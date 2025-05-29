plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.argmaxinc.whisperax"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.argmaxinc.whisperax"
        minSdk = 26
        targetSdk = 35
        versionCode = 6
        versionName = "0.1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
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

    buildFeatures {
        compose = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation(libs.core.ktx)
    implementation(libs.lifecycle.runtime.ktx)
    implementation(libs.activity.compose)

    // Material Components for XML layouts (for backward compatibility)
    implementation(libs.material)
    implementation(libs.appcompat)

    // Compose dependencies with specific versions
    val composeBom = platform(libs.compose.bom)
    implementation(composeBom)
    androidTestImplementation(composeBom)

    // Individual Compose dependencies
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)

    // Material Icons Extended
    implementation(libs.compose.material.icons.extended)

    // Navigation and ViewModel
    implementation(libs.navigation.compose)
    implementation(libs.lifecycle.viewmodel.compose)

    // For permissions handling
    implementation(libs.accompanist.permissions)

    // JSON serialization (for your existing app)
    implementation(libs.kotlinx.serialization.json)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.espresso.core)
    androidTestImplementation(libs.compose.ui.test.junit4)
    debugImplementation(libs.compose.ui.tooling)
    debugImplementation(libs.compose.ui.test.manifest)

    // To run whisperkit, include the following two sets of dependencies

    // 1 - WhisperKit API
    implementation(project(":android:whisperkit"))
    // 2 - dependencies to accelerate inference where QNN hardware is avaiable
    implementation(libs.qnn.runtime)
    implementation(libs.qnn.litert.delegate)
}
