package com.example.hearmate

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.navigation.compose.rememberNavController
import com.example.hearmate.navigation.AppNavHost
import com.example.hearmate.presentation.ui.theme.HearMateTheme
import com.example.hearmate.utils.PermissionUtils
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            Toast.makeText(this, "Microphone permission granted", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(
                this,
                "Microphone permission is required for sound detection",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Request microphone permission if not granted
        if (!PermissionUtils.hasRecordAudioPermission(this)) {
            requestPermissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
        }

        setContent {
            HearMateTheme {
                val navController = rememberNavController()
                AppNavHost(navController)
            }
        }
    }


}