package com.example.hearmate.navigation

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.example.hearmate.presentation.ui.screens.ListeningScreen
import com.example.hearmate.presentation.ui.screens.SettingsScreen
import com.example.hearmate.presentation.viewmodel.ListeningViewModel

/**
 * Composable that sets up the main navigation graph for the application.
 *
 * @param navController The NavController used for navigation.
 * @param modifier Modifier to be applied to the NavHost.
 * @param startDestination The route for the start destination of this NavHost.
 */
@Composable
fun AppNavHost(
    navController: NavHostController,
    modifier: Modifier = Modifier,
    startDestination: String = AppDestinations.LISTENING_ROUTE
) {
    NavHost(
        navController = navController,
        startDestination = startDestination,
        modifier = modifier
    ) {
        composable(route = AppDestinations.LISTENING_ROUTE) {
            val viewModel: ListeningViewModel = hiltViewModel()
            ListeningScreen(
                viewModel = viewModel,
                onSettingsClick = {
                    navController.navigate(AppDestinations.SETTINGS_ROUTE)
                }
            )
        }

        composable(route = AppDestinations.SETTINGS_ROUTE) {
            val viewModel: ListeningViewModel = hiltViewModel()
            SettingsScreen(
                viewModel = viewModel,
                onNavigateBack = { navController.popBackStack() }
            )
        }

    }
}