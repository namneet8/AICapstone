package com.example.hearmate.presentation.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.hearmate.presentation.viewmodel.ListeningViewModel
import kotlin.math.abs

/**
 * Main screen of the HearMate app.
 *
 * UI States:
 * ----------
 * 1. OFF: Toggle is off, shows status indicator as inactive
 * 2. ON: Toggle is on, audio monitoring active, pause button visible
 * 3. PAUSED: Monitoring paused, shows countdown timer and resume button
 *
 * User Actions:
 * - Toggle switch: Turn monitoring ON/OFF
 * - Pause button: Opens duration picker dialog
 * - Resume button: Immediately resumes monitoring
 * - Settings icon: Navigate to settings screen
 */
@RequiresApi(Build.VERSION_CODES.O)
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ListeningScreen(
    viewModel: ListeningViewModel,
    onSettingsClick: () -> Unit
) {
    val context = LocalContext.current

    // Collect states from ViewModel (proxied from Service)
    val isListeningEnabled by viewModel.isListeningEnabled.collectAsState()
    val isPaused by viewModel.isPaused.collectAsState()
    val pauseTimeRemaining by viewModel.pauseTimeRemaining.collectAsState()

    // Local UI state for pause duration dialog
    var showPauseDialog by remember { mutableStateOf(false) }

    // Permission launcher for microphone access
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            viewModel.startListening()
        } else {
            Toast.makeText(
                context,
                "Microphone permission is required for sound detection",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    /**
     * Handles the start listening action with permission check.
     * If permission granted, starts listening.
     * If not, requests permission first.
     */
    fun handleStartListening() {
        val hasPermission = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

        if (hasPermission) {
            viewModel.startListening()
        } else {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    // ========================================
    // Main UI Layout
    // ========================================

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("HearMate") },
                actions = {
                    IconButton(onClick = onSettingsClick) {
                        Icon(
                            imageVector = Icons.Default.Menu,
                            contentDescription = "Settings",
                            tint = MaterialTheme.colorScheme.onSurface
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                // ----------------------------------------
                // Status Indicator Area
                // ----------------------------------------
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(280.dp),
                    contentAlignment = Alignment.Center
                ) {
                    when {
                        isPaused -> {
                            // Show pause timer when paused
                            PausedStateDisplay(pauseTimeRemaining = pauseTimeRemaining)
                        }
                        else -> {
                            // Show ON/OFF status indicator
                            StatusIndicator(isActive = isListeningEnabled)
                        }
                    }
                }

                Spacer(modifier = Modifier.height(if (isPaused) 4.dp else 48.dp))

                // ----------------------------------------
                // Toggle Switch (hidden when paused)
                // ----------------------------------------
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(if (isPaused) 0.dp else 84.dp),
                    contentAlignment = Alignment.Center
                ) {
                    if (!isPaused) {
                        ToggleSwitch(
                            isEnabled = isListeningEnabled,
                            onToggle = { enabled ->
                                if (enabled) {
                                    handleStartListening()
                                } else {
                                    viewModel.stopListening()
                                }
                            },
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }

                Spacer(modifier = Modifier.height(if (isPaused) 4.dp else 24.dp))

                // ----------------------------------------
                // Action Button Area
                // ----------------------------------------
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                    contentAlignment = Alignment.Center
                ) {
                    when {
                        // Show pause button when actively listening
                        isListeningEnabled && !isPaused -> {
                            IconButton(
                                onClick = { showPauseDialog = true },
                                modifier = Modifier
                                    .size(56.dp)
                                    .clip(CircleShape)
                                    .background(MaterialTheme.colorScheme.primaryContainer)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Pause,
                                    contentDescription = "Pause Monitoring",
                                    tint = MaterialTheme.colorScheme.onPrimaryContainer,
                                    modifier = Modifier.size(28.dp)
                                )
                            }
                        }

                        // Show resume button when paused
                        isPaused -> {
                            FilledTonalButton(
                                onClick = { viewModel.resumeListening() },
                                modifier = Modifier
                                    .fillMaxWidth(0.75f)
                                    .height(56.dp)
                                    .shadow(4.dp, RoundedCornerShape(20.dp)),
                                shape = RoundedCornerShape(20.dp),
                                colors = ButtonDefaults.filledTonalButtonColors(
                                    containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                                    contentColor = MaterialTheme.colorScheme.onTertiaryContainer
                                )
                            ) {
                                Text(
                                    "Resume Now",
                                    style = MaterialTheme.typography.titleMedium
                                )
                            }
                        }

                        // Empty space when OFF
                        else -> { }
                    }
                }
            }

            // ----------------------------------------
            // Pause Duration Dialog
            // ----------------------------------------
            if (showPauseDialog) {
                PauseDurationDialog(
                    onDismiss = { showPauseDialog = false },
                    onDurationSelected = { minutes ->
                        viewModel.pauseListening(minutes)
                        showPauseDialog = false
                    }
                )
            }
        }
    }
}

// ============================================
// Status Indicator Components
// ============================================

/**
 * Visual indicator showing ON/OFF state.
 * Displays a glowing circle that changes color based on state.
 */
@Composable
private fun StatusIndicator(isActive: Boolean) {
    val statusColor = if (isActive) {
        MaterialTheme.colorScheme.primary
    } else {
        MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
    }

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Outer glow circle
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(160.dp)
                .shadow(
                    elevation = if (isActive) 12.dp else 4.dp,
                    shape = CircleShape,
                    ambientColor = statusColor.copy(alpha = 0.3f),
                    spotColor = statusColor.copy(alpha = 0.3f)
                )
                .clip(CircleShape)
                .background(
                    brush = Brush.radialGradient(
                        colors = listOf(
                            statusColor.copy(alpha = 0.3f),
                            statusColor.copy(alpha = 0.1f)
                        )
                    )
                )
        ) {
            // Inner solid circle
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(CircleShape)
                    .background(statusColor)
            )
        }
    }
}

/**
 * Display shown when monitoring is paused.
 * Shows countdown timer until auto-resume.
 */
@Composable
private fun PausedStateDisplay(pauseTimeRemaining: Long) {
    val color = MaterialTheme.colorScheme.tertiary
    val textColor = MaterialTheme.colorScheme.onSurface

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
        modifier = Modifier.fillMaxWidth()
    ) {
        // Pause indicator circle
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(120.dp)
                .clip(CircleShape)
                .background(
                    brush = Brush.radialGradient(
                        listOf(
                            color.copy(alpha = 0.3f),
                            color.copy(alpha = 0.1f)
                        )
                    )
                )
                .shadow(6.dp, CircleShape)
        ) {
            Box(
                modifier = Modifier
                    .size(60.dp)
                    .clip(CircleShape)
                    .background(color)
            )
        }

        Spacer(modifier = Modifier.height(20.dp))

        Text(
            text = "Paused",
            style = MaterialTheme.typography.headlineMedium,
            color = color
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Countdown timer display
        Row(
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = "Resumes in",
                style = MaterialTheme.typography.titleMedium,
                color = textColor.copy(alpha = 0.6f)
            )

            Spacer(modifier = Modifier.width(8.dp))

            Text(
                text = formatPauseTime(pauseTimeRemaining),
                style = MaterialTheme.typography.titleMedium.copy(
                    fontWeight = FontWeight.SemiBold
                ),
                color = textColor
            )
        }
    }
}

// ============================================
// Toggle Switch Component
// ============================================

/**
 * Styled toggle switch for turning monitoring ON/OFF.
 */
@Composable
private fun ToggleSwitch(
    isEnabled: Boolean,
    onToggle: (Boolean) -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.shadow(4.dp, RoundedCornerShape(20.dp)),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (isEnabled) {
                MaterialTheme.colorScheme.primaryContainer
            } else {
                MaterialTheme.colorScheme.surfaceVariant
            }
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = if (isEnabled) "Turn Off" else "Turn On",
                style = MaterialTheme.typography.titleLarge,
                color = if (isEnabled) {
                    MaterialTheme.colorScheme.onPrimaryContainer
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                }
            )

            Switch(
                checked = isEnabled,
                onCheckedChange = onToggle,
                colors = SwitchDefaults.colors(
                    checkedThumbColor = MaterialTheme.colorScheme.onPrimary,
                    checkedTrackColor = MaterialTheme.colorScheme.primary,
                    checkedBorderColor = MaterialTheme.colorScheme.primary,
                    uncheckedThumbColor = MaterialTheme.colorScheme.outline,
                    uncheckedTrackColor = MaterialTheme.colorScheme.surfaceVariant,
                    uncheckedBorderColor = MaterialTheme.colorScheme.outline
                )
            )
        }
    }
}

// ============================================
// Pause Duration Dialog
// ============================================

/**
 * Dialog for selecting pause duration.
 * Uses wheel pickers for hours and minutes.
 */
@Composable
private fun PauseDurationDialog(
    onDismiss: () -> Unit,
    onDurationSelected: (Int) -> Unit
) {
    var selectedHours by remember { mutableIntStateOf(0) }
    var selectedMinutes by remember { mutableIntStateOf(15) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                "Pause Duration",
                style = MaterialTheme.typography.headlineSmall
            )
        },
        text = {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    "Select how long to pause monitoring",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(24.dp))

                // Time picker wheels
                Row(
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    TimeWheelPicker(
                        items = (0..12).toList(),
                        selectedItem = selectedHours,
                        onItemSelected = { selectedHours = it },
                        label = "h"
                    )

                    Spacer(modifier = Modifier.width(16.dp))

                    TimeWheelPicker(
                        items = (0..59 step 5).toList(),
                        selectedItem = selectedMinutes,
                        onItemSelected = { selectedMinutes = it },
                        label = "m"
                    )
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = {
                    val totalMinutes = (selectedHours * 60) + selectedMinutes
                    if (totalMinutes > 0) {
                        onDurationSelected(totalMinutes)
                    }
                },
                enabled = (selectedHours > 0 || selectedMinutes > 0)
            ) {
                Text("Confirm")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}

/**
 * Scrollable wheel picker for time selection.
 */
@Composable
private fun TimeWheelPicker(
    items: List<Int>,
    selectedItem: Int,
    onItemSelected: (Int) -> Unit,
    label: String,
    modifier: Modifier = Modifier
) {
    val listState = rememberLazyListState(
        initialFirstVisibleItemIndex = maxOf(0, items.indexOf(selectedItem).coerceAtLeast(0) - 2)
    )

    // Track which item is centered in the viewport
    val centerIndex by remember {
        derivedStateOf {
            val layoutInfo = listState.layoutInfo
            val viewportCenter = layoutInfo.viewportStartOffset + layoutInfo.viewportSize.height / 2

            layoutInfo.visibleItemsInfo
                .minByOrNull { itemInfo ->
                    val itemCenter = itemInfo.offset + itemInfo.size / 2
                    abs(itemCenter - viewportCenter)
                }?.index ?: items.indexOf(selectedItem)
        }
    }

    // Update selected item when scroll settles on new center
    LaunchedEffect(centerIndex) {
        if (centerIndex in items.indices && items[centerIndex] != selectedItem) {
            onItemSelected(items[centerIndex])
        }
    }

    Box(
        modifier = modifier
            .width(80.dp)
            .height(150.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
    ) {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            state = listState,
            contentPadding = PaddingValues(vertical = 50.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            items(items.size) { index ->
                val item = items[index]
                val isSelected = item == selectedItem

                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(50.dp)
                        .background(
                            if (isSelected) {
                                MaterialTheme.colorScheme.primaryContainer
                            } else {
                                Color.Transparent
                            }
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "%02d$label".format(item),
                        style = MaterialTheme.typography.titleLarge,
                        color = if (isSelected) {
                            MaterialTheme.colorScheme.onPrimaryContainer
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                        }
                    )
                }
            }
        }
    }
}

// ============================================
// Utility Functions
// ============================================

/**
 * Formats seconds into MM:SS format for timer display.
 */
private fun formatPauseTime(seconds: Long): String {
    val minutes = seconds / 60
    val secs = seconds % 60
    return "%02d:%02d".format(minutes, secs)
}