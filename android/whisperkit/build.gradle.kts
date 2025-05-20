import com.vanniktech.maven.publish.SonatypeHost
import io.gitlab.arturbosch.detekt.Detekt
import io.gitlab.arturbosch.detekt.DetektCreateBaselineTask

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.vanniktech.maven.publish)
    alias(libs.plugins.detekt)
}

android {
    namespace = "com.argmaxinc.whisperkit"
    compileSdk = 35

    defaultConfig {
        minSdk = 26

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")

        // Configure cmake arguments
        externalNativeBuild {
            cmake {
                arguments("-DJNI=1", "-DTENSORFLOW_SOURCE_DIR=${rootProject.projectDir}/external/tensorflow")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }
}

dependencies {
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.ktor.client.core)
    implementation(libs.ktor.client.cio)
    implementation(libs.ktor.client.content.negotiation)
    implementation(libs.ktor.serialization.kotlinx.json)
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.mockk)
    testImplementation(libs.turbine)
    testImplementation(libs.ktor.client.mock)
}

mavenPublishing {

    coordinates("com.argmaxinc", "whisperkit", "0.3.0")
    pom {
        name.set("WhisperKit")
        description.set("On-device Speech Recognition for Android")
        inceptionYear.set("2025")
        url.set("https://github.com/argmaxinc/WhisperKitAndroid")

        licenses {
            license {
                name.set("MIT")
                url.set("https://opensource.org/licenses/MIT")
                distribution.set("repo")
            }
        }

        developers {
            developer {
                id.set("argmaxinc")
                name.set("Argmax")
                url.set("https://github.com/argmaxinc/")
            }
        }

        scm {
            url.set("https://github.com/argmaxinc/WhisperKitAndroid")
            connection.set("scm:git:git://github.com/argmaxinc/WhisperKitAndroid.git")
            developerConnection.set("scm:git:ssh://git@github.com:argmaxinc/WhisperKitAndroid.git")
        }
    }
    signAllPublications()
    publishToMavenCentral(SonatypeHost.CENTRAL_PORTAL)
}

detekt {
    buildUponDefaultConfig = true
    config.setFrom(files("${rootProject.projectDir}/android/config/detekt.yml"))
    baseline = file("${rootProject.projectDir}/android/whisperkit/detekt-baseline.xml")
}

tasks.withType<Detekt>().configureEach {
    reports {
        html.required.set(true)
    }
}

tasks.withType<Detekt>().configureEach {
    jvmTarget = "1.8"
}
tasks.withType<DetektCreateBaselineTask>().configureEach {
    jvmTarget = "1.8"
}
