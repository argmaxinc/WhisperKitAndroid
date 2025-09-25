package com.argmaxinc.whisperkit.huggingface

import app.cash.turbine.test
import com.argmaxinc.whisperkit.huggingface.HuggingFaceApiConfig.Companion.DEFAULT_MAX_RETRY
import io.ktor.client.HttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.http.HttpStatusCode
import io.ktor.http.headers
import io.ktor.serialization.kotlinx.json.json
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import kotlin.time.Duration.Companion.seconds

@OptIn(ExperimentalCoroutinesApi::class)
internal class KtorHuggingFaceApiImplTest {
    private lateinit var api: KtorHuggingFaceApiImpl
    private lateinit var testDir: File
    private lateinit var mockLogger: HuggingFaceLogger
    private lateinit var httpClient: HttpClient
    private val mockClient: KtorHuggingFaceClient = mockk()

    enum class EngineBehavior {
        SUCCESS,
        CORRUPTED_FILE,
        RETRY_SUCCESS,
    }

    private var retryCount = 0

    private fun createEngine(behavior: EngineBehavior) =
        MockEngine { request ->
            when {
                request.url.toString().contains("/api/models/test-repo") -> {
                    respond(
                        content =
                        """
                            {
                                "siblings": [
                                    {"rfilename": "test1.txt", "size": 12},
                                    {"rfilename": "test2.txt", "size": 10000}
                                ]
                            }
                        """.trimIndent(),
                        status = HttpStatusCode.OK,
                        headers =
                        headers {
                            append("Content-Type", "application/json")
                        },
                    )
                }

                request.url.toString().contains("/resolve/main/test1.txt") -> {
                    respond(
                        content = if (request.method.value == "HEAD") "" else "x".repeat(12),
                        status = HttpStatusCode.OK,
                        headers =
                        headers {
                            append("Content-Type", "application/octet-stream")
                            append("Content-Length", "12")
                        },
                    )
                }
                // test2.txt has 10000 bytes
                request.url.toString().contains("/resolve/main/test2.txt") -> {
                    respond(
                        content =
                        if (request.method.value == "HEAD") {
                            ""
                        } else {
                            when (behavior) {
                                EngineBehavior.SUCCESS -> "x".repeat(10000)
                                EngineBehavior.CORRUPTED_FILE -> "corrupted"
                                EngineBehavior.RETRY_SUCCESS -> {
                                    retryCount++
                                    if (retryCount > 1) {
                                        "x".repeat(10000)
                                    } else {
                                        "corrupted"
                                    }
                                }
                            }
                        },
                        status = HttpStatusCode.OK,
                        headers =
                        headers {
                            append("Content-Type", "application/octet-stream")
                            append("Content-Length", "10000")
                        },
                    )
                }

                else -> {
                    respond(
                        content = "{}",
                        status = HttpStatusCode.BadRequest,
                    )
                }
            }
        }

    private fun setupWithEngine(
        engine: MockEngine,
        ioDispatcher: CoroutineDispatcher,
    ) {
        testDir = File("build/test-downloads").apply { mkdirs() }
        mockLogger = mockk(relaxed = true)
        httpClient =
            HttpClient(engine) {
                install(ContentNegotiation) {
                    json(
                        Json {
                            ignoreUnknownKeys = true
                            isLenient = true
                        },
                    )
                }
            }
        every { mockClient.httpClient } returns httpClient

        api =
            KtorHuggingFaceApiImpl(
                config =
                HuggingFaceApiConfig(
                    retryCount = DEFAULT_MAX_RETRY,
                    logger = mockLogger,
                ),
                client = mockClient,
                ioDispatcher = ioDispatcher,
            )
    }

    @After
    fun cleanup() {
        testDir.deleteRecursively()
    }

    @Before
    fun setup() {
        retryCount = 0
    }

    @Test
    fun `test successful download with progress`() =
        runTest {
            setupWithEngine(
                createEngine(EngineBehavior.SUCCESS),
                UnconfinedTestDispatcher(testScheduler),
            )
            val repo = Repo("test-repo", RepoType.MODELS)

            // download test1.txt and test2.txt
            api.snapshot(repo, "main", listOf("test*"), testDir).test {
                // Verify first progress (after test1.txt)
                val firstProgress = awaitItem()
                assertTrue(firstProgress.fractionCompleted < 1.0f)
                verify {
                    mockLogger.info(
                        "files to download: [test1.txt, test2.txt], " +
                            "totalBytes: 10012, totalFiles: 2",
                    )
                }
                // Verify second progress (first chunk of test2.txt)
                val secondProgress = awaitItem()
                assertTrue(secondProgress.fractionCompleted < 1.0f)

                // Verify third progress (second chunk of test2.txt)
                val thirdProgress = awaitItem()
                assertTrue(thirdProgress.isDone)
                // Verify completion
                awaitComplete()

                verify { mockLogger.info("ALL Files downloaded! [test1.txt, test2.txt]") }
            }

            // Verify files were downloaded
            val downloadedFile1 = File(testDir, "test1.txt")
            val downloadedFile2 = File(testDir, "test2.txt")
            assertTrue(downloadedFile1.exists())
            assertTrue(downloadedFile2.exists())
            assertEquals(12L, downloadedFile1.length())
            assertEquals(10000L, downloadedFile2.length())
        }

    @Test
    fun `test download fails after max retries`() =
        runTest {
            setupWithEngine(
                createEngine(EngineBehavior.CORRUPTED_FILE),
                UnconfinedTestDispatcher(testScheduler),
            )
            val repo = Repo("test-repo", RepoType.MODELS)
            api.snapshot(repo, "main", listOf("test2.txt"), testDir).test(timeout = 10.seconds) {
                awaitItem() // first progress
                val error = awaitError() // exception thrown
                assertTrue(error is IllegalStateException)
                assertTrue(error.message?.contains("Failed to download test2.txt after 3 attempts") == true)

                // Verify file was deleted after size mismatch
                val downloadedFile = File(testDir, "test2.txt")
                assertTrue(!downloadedFile.exists())

                // await for all completion before verifying logs
                verify { mockLogger.info("Retry attempt 0 for test2.txt") }
                verify { mockLogger.error(any(), "Error during download for test2.txt (attempt 0)") }
                verify { mockLogger.info("Wait and retry downloading test2.txt...") }

                verify { mockLogger.info("Retry attempt 1 for test2.txt") }
                verify { mockLogger.error(any(), "Error during download for test2.txt (attempt 1)") }
                verify { mockLogger.info("Wait and retry downloading test2.txt...") }

                verify { mockLogger.info("Retry attempt 2 for test2.txt") }
                verify { mockLogger.error(any(), "Error during download for test2.txt (attempt 2)") }
                verify { mockLogger.info("Wait and retry downloading test2.txt...") }
                verify { mockLogger.error("Failed to download test2.txt after 3 attempts") }
            }
        }

    @Test
    fun `test succeeds after retry and progress reporting is monotonic`() =
        runTest {
            setupWithEngine(
                createEngine(EngineBehavior.RETRY_SUCCESS),
                UnconfinedTestDispatcher(testScheduler),
            )
            val repo = Repo("test-repo", RepoType.MODELS)
            val progressValues = mutableListOf<Float>()

            api.snapshot(repo, "main", listOf("test2.txt"), testDir).test {
                val firstProgress = awaitItem()
                progressValues.add(firstProgress.fractionCompleted)
                verify { mockLogger.info("Retry attempt 0 for test2.txt") }
                verify { mockLogger.error(any(), "Error during download for test2.txt (attempt 0)") }
                verify { mockLogger.info("Wait and retry downloading test2.txt...") }

                val secondProgress = awaitItem()
                progressValues.add(secondProgress.fractionCompleted)
                verify { mockLogger.info("Retry attempt 1 for test2.txt") }

                val thirdProgress = awaitItem()
                progressValues.add(thirdProgress.fractionCompleted)
                verify { mockLogger.info("ALL Files downloaded! [test2.txt]") }

                awaitComplete()

                // Verify progress never decreases
                assertTrue(progressValues.isNotEmpty())
                for (i in 1 until progressValues.size) {
                    assertTrue(
                        "Progress should never decrease",
                        progressValues[i] >= progressValues[i - 1],
                    )
                }
            }
        }

    @Test
    fun `test getFileNames returns filtered list`() =
        runTest {
            setupWithEngine(
                createEngine(EngineBehavior.SUCCESS),
                UnconfinedTestDispatcher(testScheduler),
            )
            val repo = Repo("test-repo", RepoType.MODELS)
            val globFilters = listOf("test*")
            assertEquals(listOf("test1.txt", "test2.txt"), api.getFileNames(repo, "main", globFilters))
        }

    @Test
    fun `test getModelInfo returns correct model info`() =
        runTest {
            setupWithEngine(
                createEngine(EngineBehavior.SUCCESS),
                UnconfinedTestDispatcher(testScheduler),
            )
            val repo = Repo("test-repo", RepoType.MODELS)
            val modelInfo = api.getModelInfo(repo)
            assertEquals(2, modelInfo.siblings!!.size)
            assertEquals("test1.txt", modelInfo.siblings!![0].rfilename)
            assertEquals("test2.txt", modelInfo.siblings!![1].rfilename)
        }

    @Test
    fun `test getModelInfo throws IllegalArgumentException for non-model repo`() =
        runTest {
            setupWithEngine(
                createEngine(EngineBehavior.SUCCESS),
                UnconfinedTestDispatcher(testScheduler),
            )
            val repo = Repo("test-repo", RepoType.DATASETS)
            try {
                api.getModelInfo(repo)
                assertTrue("Should have thrown IllegalArgumentException", false)
            } catch (e: IllegalArgumentException) {
                assertTrue(e.message?.contains("needs to have type RepoType.MODELS") == true)
            }
        }

    @Test
    fun `test snapshot emits completed progress when no files to download`() =
        runTest {
            setupWithEngine(
                createEngine(EngineBehavior.SUCCESS),
                UnconfinedTestDispatcher(testScheduler),
            )
            val repo = Repo("test-repo", RepoType.MODELS)

            api.snapshot(repo, "main", listOf("nonexistent*"), testDir).test {
                // Should emit a single progress with 1.0f
                val progress = awaitItem()
                assertTrue(progress.isDone)
                verify { mockLogger.info("No files to download, finish immediately, for Repo(test-repo, main) and glob filters: [nonexistent*]") }
                awaitComplete()
            }
        }

    @Test
    fun `test snapshot handles empty file list correctly`() =
        runTest {
            setupWithEngine(
                createEngine(EngineBehavior.SUCCESS),
                UnconfinedTestDispatcher(testScheduler),
            )
            val repo = Repo("test-repo", RepoType.MODELS)
            api.snapshot(repo, "main", emptyList(), testDir).test {
                val progress = awaitItem()
                awaitComplete()
                assertTrue(progress.isDone)
                verify { mockLogger.info("No files to download, finish immediately, for Repo(test-repo, main) and glob filters: []") }
            }
        }

    @Test
    fun `test snapshot creates parent directories`() =
        runTest {
            setupWithEngine(
                createEngine(EngineBehavior.SUCCESS),
                UnconfinedTestDispatcher(testScheduler),
            )
            val repo = Repo("test-repo", RepoType.MODELS)
            val nestedDir = File(testDir, "nested/path")
            api.snapshot(repo, "main", listOf("test1.txt"), nestedDir).test {
                awaitItem() // progress
                awaitComplete()
            }
            val downloadedFile = File(nestedDir, "test1.txt")
            assertTrue(downloadedFile.exists())
            assertTrue(downloadedFile.parentFile?.exists() == true)
        }
}
