package com.example.hearmate.presentation.ui.screens

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.hearmate.presentation.viewmodel.ListeningViewModel

/**
 * Settings screen that allows users to configure alert preferences
 * for emergency sound detections.
 *
 * Features:
 * - Toggle alerts on/off for each emergency sound type
 * - Settings are persisted using SharedPreferences
 * - Real-time updates when toggles are changed
 */
@RequiresApi(Build.VERSION_CODES.O)
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onNavigateBack: () -> Unit,
    viewModel: ListeningViewModel,
    modifier: Modifier = Modifier
) {
    // Get all emergency sounds from ViewModel
    val emergencySounds = viewModel.getEmergencySounds()

    // Collect alert preferences as state
    val alertPreferences by viewModel.vibrationPreferences.collectAsState()

    Scaffold(
        modifier = modifier.fillMaxSize(),
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer,
                    navigationIconContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .padding(innerPadding)
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Section title for alert options
            Text(
                text = "Alert Options",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(bottom = 8.dp)
            )

            Text(
                text = "Enable or disable alerts for each emergency sound type",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(8.dp))

            // List of emergency sounds with toggle switches
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(emergencySounds) { soundLabel ->
                    EmergencySoundToggleCard(
                        soundLabel = soundLabel,
                        isEnabled = alertPreferences[soundLabel] ?: true,
                        onToggle = { enabled ->
                            viewModel.setVibrationEnabled(soundLabel, enabled)
                        }
                    )
                }
            }
        }
    }
}

/**
 * Card component that displays a single emergency sound with a toggle switch
 * to enable/disable alerts for that specific sound type.
 *
 * @param soundLabel The name of the emergency sound (e.g., "Gunshot", "Siren")
 * @param isEnabled Current alert state for this sound
 * @param onToggle Callback invoked when the switch is toggled
 */
@Composable
private fun EmergencySoundToggleCard(
    soundLabel: String,
    isEnabled: Boolean,
    onToggle: (Boolean) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Display the sound label with formatted text (replace underscores with spaces)
            Text(
                text = soundLabel.replace("_", " "),
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.weight(1f)
            )

            // Toggle switch for enabling/disabling alerts
            Switch(
                checked = isEnabled,
                onCheckedChange = onToggle,
                colors = SwitchDefaults.colors(
                    checkedThumbColor = MaterialTheme.colorScheme.primary,
                    checkedTrackColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    }
}