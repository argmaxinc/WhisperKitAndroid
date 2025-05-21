plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.serialization) apply false
    alias(libs.plugins.detekt) apply false
    alias(libs.plugins.spotless)
}

spotless {
    kotlin {
        target("android/**/*.kt")
        targetExclude("**/build/**", "**/generated/**")
        ktlint().editorConfigOverride(
            mapOf(
                // Disable naming rules
                "ktlint_standard_function-naming" to "disabled",
                "ktlint_standard_property-naming" to "disabled",
            ),
        )
        trimTrailingWhitespace()
        endWithNewline()
    }

    kotlinGradle {
        target("**/*.gradle.kts")
        ktlint()
        trimTrailingWhitespace()
        endWithNewline()
    }

    cpp {
        target(
            "jni/**/*.cpp", "jni/**/*.h", "jni/**/*.c", "jni/**/*.hpp", "jni/**/*.cc",
            "cli/**/*.cpp", "cli/**/*.h", "cli/**/*.c", "cli/**/*.hpp", "cli/**/*.cc",
            "cpp/**/*.cpp", "cpp/**/*.h", "cpp/**/*.c", "cpp/**/*.hpp", "cpp/**/*.cc",
        )
        targetExclude("**/build/**", "**/external/**")
        clangFormat("11.1.0")
    }
}
