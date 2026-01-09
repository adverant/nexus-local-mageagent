import Cocoa
import UserNotifications
import IOKit

// MARK: - MageAgent Menu Bar Application Delegate
// Production-grade macOS menu bar application for MageAgent server management
// Implements NSMenuItemValidation for proper menu item state management
// Includes Activity Monitor-style system pressure indicators

final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuItemValidation {

    // MARK: - Properties

    /// Status bar item - retained for entire app lifecycle
    private var statusItem: NSStatusItem!

    /// Main menu - stored as strong reference
    private var menu: NSMenu!

    /// Status checking timer
    private var statusTimer: Timer?

    /// System pressure monitoring timer (faster updates)
    private var pressureTimer: Timer?

    /// Server status for menu item state management
    private var isServerRunning: Bool = false

    /// System pressure levels
    private var memoryPressure: SystemPressure = .nominal
    private var cpuUsage: Double = 0.0
    private var gpuUsage: Double = 0.0

    // MARK: - System Pressure Types

    enum SystemPressure: String {
        case nominal = "Normal"
        case warning = "Warning"
        case critical = "Critical"

        var color: NSColor {
            switch self {
            case .nominal: return .systemGreen
            case .warning: return .systemYellow
            case .critical: return .systemRed
            }
        }

        var indicator: String {
            switch self {
            case .nominal: return "●"
            case .warning: return "●"
            case .critical: return "●"
            }
        }
    }

    // MARK: - Menu Item Tags (for identification)

    private enum MenuItemTag: Int {
        case status = 100
        case models = 200
        case startServer = 300
        case stopServer = 301
        case restartServer = 302
        case warmupModels = 303
        case openDocs = 400
        case viewLogs = 401
        case runTest = 402
        case settings = 500
        case systemPressure = 600
        case memoryDetail = 601
        case cpuDetail = 602
        case gpuDetail = 603
    }

    // MARK: - Configuration

    private struct Config {
        static let mageagentScript = "\(NSHomeDirectory())/.claude/scripts/mageagent-server.sh"
        static let mageagentURL = "http://localhost:3457"
        static let logFile = "\(NSHomeDirectory())/.claude/debug/mageagent.log"
        static let debugLogFile = "\(NSHomeDirectory())/.claude/debug/mageagent-menubar-debug.log"
        static let iconPath = "\(NSHomeDirectory())/.claude/mageagent-menubar/icons/icon_18x18@2x.png"
        static let statusCheckInterval: TimeInterval = 10.0
        static let requestTimeout: TimeInterval = 2.0
    }

    // MARK: - Model and Pattern Definitions

    private struct ModelInfo {
        let modelId: String
        let displayName: String
        let memorySize: String
    }

    private struct PatternInfo {
        let patternId: String
        let displayName: String
        let requiredModels: [String]  // Model IDs required for this pattern
        let description: String
    }

    /// Available models that can be loaded
    private let availableModels: [ModelInfo] = [
        ModelInfo(modelId: "mageagent:primary", displayName: "Qwen-72B Q8 (77GB) - Reasoning", memorySize: "77GB"),
        ModelInfo(modelId: "mageagent:tools", displayName: "Hermes-3 8B Q8 (9GB) - Tool Calling", memorySize: "9GB"),
        ModelInfo(modelId: "mageagent:validator", displayName: "Qwen-Coder 7B (5GB) - Fast Validation", memorySize: "5GB"),
        ModelInfo(modelId: "mageagent:competitor", displayName: "Qwen-Coder 32B (18GB) - Coding", memorySize: "18GB")
    ]

    /// Available orchestration patterns with their required models
    private let availablePatterns: [PatternInfo] = [
        PatternInfo(
            patternId: "mageagent:auto",
            displayName: "auto",
            requiredModels: ["mageagent:validator"],  // Uses validator for classification, loads others on demand
            description: "Intelligent task routing - classifies task and routes to best model"
        ),
        PatternInfo(
            patternId: "mageagent:execute",
            displayName: "execute",
            requiredModels: ["mageagent:primary", "mageagent:tools"],
            description: "Real tool execution - ReAct loop with Qwen-72B + Hermes tool calling"
        ),
        PatternInfo(
            patternId: "mageagent:hybrid",
            displayName: "hybrid",
            requiredModels: ["mageagent:primary", "mageagent:tools"],
            description: "Reasoning + tools - Qwen-72B for thinking, Hermes for tool extraction"
        ),
        PatternInfo(
            patternId: "mageagent:validated",
            displayName: "validated",
            requiredModels: ["mageagent:primary", "mageagent:validator"],
            description: "Generate + validate - Qwen-72B generates, 7B validates and revises"
        ),
        PatternInfo(
            patternId: "mageagent:compete",
            displayName: "compete",
            requiredModels: ["mageagent:primary", "mageagent:competitor", "mageagent:validator"],
            description: "Multi-model judge - 72B + 32B generate, 7B judges best response"
        )
    ]

    /// Track which models are currently loaded
    private var loadedModels: Set<String> = []

    /// Currently selected pattern
    private var selectedPattern: String = "mageagent:auto"

    /// System memory info cache
    private var lastMemoryInfo: (used: UInt64, total: UInt64, pressure: String) = (0, 0, "nominal")

    /// Previous CPU ticks for delta calculation
    private var previousCPUTicks: (user: UInt64, system: UInt64, idle: UInt64, nice: UInt64) = (0, 0, 0, 0)

    // MARK: - Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        debugLog("Application launched - initializing MageAgent Menu Bar")

        // Configure as menu bar only app (no dock icon)
        NSApp.setActivationPolicy(.accessory)

        // Request notification permissions
        requestNotificationPermission()

        // Initialize UI components
        setupStatusItem()
        setupMenu()

        // Start monitoring server status
        checkServerStatus()
        startStatusTimer()

        // Start system pressure monitoring (every 2 seconds)
        startPressureTimer()

        debugLog("Application initialization complete")
    }

    func applicationWillTerminate(_ notification: Notification) {
        debugLog("Application terminating - cleaning up")
        statusTimer?.invalidate()
        statusTimer = nil
        pressureTimer?.invalidate()
        pressureTimer = nil
    }

    // MARK: - UI Setup

    private func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        guard let button = statusItem.button else {
            debugLog("ERROR: Failed to get status item button")
            return
        }

        // Load custom icon or fall back to system symbol
        if let iconImage = NSImage(contentsOfFile: Config.iconPath) {
            iconImage.size = NSSize(width: 25, height: 25)
            iconImage.isTemplate = true  // Adapts to light/dark mode
            button.image = iconImage
            debugLog("Custom icon loaded successfully")
        } else if let symbolImage = NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "MageAgent") {
            symbolImage.isTemplate = true
            button.image = symbolImage
            debugLog("Using fallback system symbol")
        } else {
            // Ultimate fallback - text
            button.title = "MA"
            debugLog("Using text fallback for status item")
        }

        button.toolTip = "MageAgent - Multi-Model AI Orchestration"
    }

    private func setupMenu() {
        menu = NSMenu()
        menu.autoenablesItems = false  // We manage enabled state manually

        // Status display (non-interactive)
        let statusMenuItem = NSMenuItem(title: "Status: Checking...", action: nil, keyEquivalent: "")
        statusMenuItem.tag = MenuItemTag.status.rawValue
        statusMenuItem.isEnabled = false
        menu.addItem(statusMenuItem)

        menu.addItem(NSMenuItem.separator())

        // System Pressure Section - Activity Monitor style
        // IMPORTANT: Use displayOnlyAction instead of nil/disabled to prevent macOS from dimming text
        let pressureHeader = NSMenuItem(title: "System Resources", action: #selector(displayOnlyAction(_:)), keyEquivalent: "")
        pressureHeader.tag = MenuItemTag.systemPressure.rawValue
        pressureHeader.target = self
        pressureHeader.isEnabled = true  // Keep enabled for full opacity text

        // Create attributed string with colored indicator
        let pressureAttr = NSMutableAttributedString(string: "● ", attributes: [.foregroundColor: NSColor.systemGreen])
        pressureAttr.append(NSAttributedString(string: "System Resources", attributes: [
            .foregroundColor: NSColor.black,
            .font: NSFont.systemFont(ofSize: 13, weight: .bold)
        ]))
        pressureHeader.attributedTitle = pressureAttr

        menu.addItem(pressureHeader)

        // Memory detail item - enabled with no-op action for full opacity
        let memoryItem = NSMenuItem(title: "  Memory: Checking...", action: #selector(displayOnlyAction(_:)), keyEquivalent: "")
        memoryItem.tag = MenuItemTag.memoryDetail.rawValue
        memoryItem.target = self
        memoryItem.isEnabled = true
        menu.addItem(memoryItem)

        // CPU detail item - enabled with no-op action for full opacity
        let cpuItem = NSMenuItem(title: "  CPU: Checking...", action: #selector(displayOnlyAction(_:)), keyEquivalent: "")
        cpuItem.tag = MenuItemTag.cpuDetail.rawValue
        cpuItem.target = self
        cpuItem.isEnabled = true
        menu.addItem(cpuItem)

        // GPU/Metal detail item - enabled with no-op action for full opacity
        let gpuItem = NSMenuItem(title: "  GPU/Metal: Checking...", action: #selector(displayOnlyAction(_:)), keyEquivalent: "")
        gpuItem.tag = MenuItemTag.gpuDetail.rawValue
        gpuItem.target = self
        gpuItem.isEnabled = true
        menu.addItem(gpuItem)

        menu.addItem(NSMenuItem.separator())

        // Server control section
        addMenuItem(title: "Start Server", action: #selector(startServerAction(_:)),
                   keyEquivalent: "s", tag: .startServer)
        addMenuItem(title: "Stop Server", action: #selector(stopServerAction(_:)),
                   keyEquivalent: "", tag: .stopServer)
        addMenuItem(title: "Restart Server", action: #selector(restartServerAction(_:)),
                   keyEquivalent: "r", tag: .restartServer)
        addMenuItem(title: "Warmup Models", action: #selector(warmupModelsAction(_:)),
                   keyEquivalent: "w", tag: .warmupModels)

        menu.addItem(NSMenuItem.separator())

        // Models submenu - clickable to load individual models
        let modelsItem = NSMenuItem(title: "Load Models", action: nil, keyEquivalent: "")
        modelsItem.tag = MenuItemTag.models.rawValue
        let modelsSubmenu = NSMenu()

        // Add header
        let headerItem = NSMenuItem(title: "Click to load into memory:", action: nil, keyEquivalent: "")
        headerItem.isEnabled = false
        modelsSubmenu.addItem(headerItem)
        modelsSubmenu.addItem(NSMenuItem.separator())

        // Add each model as a clickable item
        for (index, model) in availableModels.enumerated() {
            let item = NSMenuItem(title: model.displayName, action: #selector(loadModelAction(_:)), keyEquivalent: "")
            item.target = self
            item.tag = 1000 + index  // Tags starting at 1000 for models
            item.representedObject = model.modelId
            modelsSubmenu.addItem(item)
        }

        modelsSubmenu.addItem(NSMenuItem.separator())

        // Load all models option
        let loadAllItem = NSMenuItem(title: "Load All Models", action: #selector(warmupModelsAction(_:)), keyEquivalent: "w")
        loadAllItem.target = self
        loadAllItem.keyEquivalentModifierMask = [.command]
        modelsSubmenu.addItem(loadAllItem)

        modelsItem.submenu = modelsSubmenu
        menu.addItem(modelsItem)

        // Patterns submenu - each pattern shows required models as a submenu
        let patternsItem = NSMenuItem(title: "Patterns", action: nil, keyEquivalent: "")
        let patternsSubmenu = NSMenu()

        // Add header
        let patternHeader = NSMenuItem(title: "Select pattern (click to load required models):", action: nil, keyEquivalent: "")
        patternHeader.isEnabled = false
        patternsSubmenu.addItem(patternHeader)
        patternsSubmenu.addItem(NSMenuItem.separator())

        // Add each pattern with its own submenu showing required models
        for (index, pattern) in availablePatterns.enumerated() {
            let patternItem = NSMenuItem(title: pattern.displayName, action: nil, keyEquivalent: "")
            patternItem.tag = 2000 + index

            // Create submenu for this pattern
            let patternDetailsSubmenu = NSMenu()

            // Description header
            let descItem = NSMenuItem(title: pattern.description, action: nil, keyEquivalent: "")
            descItem.isEnabled = false
            patternDetailsSubmenu.addItem(descItem)
            patternDetailsSubmenu.addItem(NSMenuItem.separator())

            // Required models header
            let reqHeader = NSMenuItem(title: "Required Models:", action: nil, keyEquivalent: "")
            reqHeader.isEnabled = false
            patternDetailsSubmenu.addItem(reqHeader)

            // List each required model
            for modelId in pattern.requiredModels {
                if let modelInfo = availableModels.first(where: { $0.modelId == modelId }) {
                    let modelItem = NSMenuItem(title: "  • \(modelInfo.displayName)", action: nil, keyEquivalent: "")
                    modelItem.isEnabled = false
                    patternDetailsSubmenu.addItem(modelItem)
                }
            }

            patternDetailsSubmenu.addItem(NSMenuItem.separator())

            // "Use this pattern" action - loads all required models
            let usePatternItem = NSMenuItem(title: "Use \(pattern.displayName) Pattern", action: #selector(selectPatternAction(_:)), keyEquivalent: "")
            usePatternItem.target = self
            usePatternItem.representedObject = pattern
            patternDetailsSubmenu.addItem(usePatternItem)

            patternItem.submenu = patternDetailsSubmenu
            patternsSubmenu.addItem(patternItem)
        }

        patternsItem.submenu = patternsSubmenu
        menu.addItem(patternsItem)

        menu.addItem(NSMenuItem.separator())

        // Utility actions
        addMenuItem(title: "Open API Docs", action: #selector(openDocsAction(_:)),
                   keyEquivalent: "d", tag: .openDocs)
        addMenuItem(title: "View Logs", action: #selector(viewLogsAction(_:)),
                   keyEquivalent: "l", tag: .viewLogs)
        addMenuItem(title: "Run Test", action: #selector(runTestAction(_:)),
                   keyEquivalent: "t", tag: .runTest)

        menu.addItem(NSMenuItem.separator())

        // Settings
        addMenuItem(title: "Settings...", action: #selector(showSettingsAction(_:)),
                   keyEquivalent: ",", tag: .settings)

        menu.addItem(NSMenuItem.separator())

        // Quit
        let quitItem = NSMenuItem(title: "Quit MageAgent Menu", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        menu.addItem(quitItem)

        // Assign menu to status item
        statusItem.menu = menu

        debugLog("Menu setup complete with \(menu.items.count) items")
    }

    /// Helper to create menu items with proper target/action binding
    private func addMenuItem(title: String, action: Selector, keyEquivalent: String, tag: MenuItemTag) {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: keyEquivalent)
        item.target = self
        item.tag = tag.rawValue
        if !keyEquivalent.isEmpty {
            item.keyEquivalentModifierMask = [.command]
        }
        menu.addItem(item)
    }

    // MARK: - NSMenuItemValidation

    /// Called by AppKit to determine if menu items should be enabled
    func validateMenuItem(_ menuItem: NSMenuItem) -> Bool {
        guard let tag = MenuItemTag(rawValue: menuItem.tag) else {
            return true  // Allow untagged items
        }

        switch tag {
        case .startServer:
            return !isServerRunning
        case .stopServer, .restartServer, .warmupModels:
            return isServerRunning
        case .openDocs:
            return isServerRunning
        case .viewLogs, .runTest, .settings:
            return true
        default:
            return true
        }
    }

    // MARK: - No-Op Action for Display-Only Menu Items
    // macOS automatically dims disabled menu items, so we use enabled items with a no-op action
    // to keep text at full opacity while remaining non-interactive

    @objc func displayOnlyAction(_ sender: NSMenuItem) {
        // Intentionally empty - this keeps menu items enabled (full opacity)
        // while not performing any action when clicked
    }

    // MARK: - Server Control Actions

    @objc func startServerAction(_ sender: NSMenuItem) {
        debugLog("Start Server action triggered")
        executeServerCommand("start", successMessage: "Server started on port 3457",
                           failureMessage: "Failed to start server")
    }

    @objc func stopServerAction(_ sender: NSMenuItem) {
        debugLog("Stop Server action triggered")
        executeServerCommand("stop", successMessage: "Server stopped",
                           failureMessage: "Failed to stop server")
    }

    @objc func restartServerAction(_ sender: NSMenuItem) {
        debugLog("Restart Server action triggered")
        executeServerCommand("restart", successMessage: "Server restarted",
                           failureMessage: "Failed to restart server")
    }

    @objc func warmupModelsAction(_ sender: NSMenuItem) {
        debugLog("Warmup Models action triggered")
        sendNotification(title: "MageAgent", body: "Warming up models - this may take a few minutes...")

        // Models to warm up - each will load into GPU/RAM
        let modelsToWarmup = [
            ("mageagent:primary", "Qwen-72B Q8 (77GB)"),
            ("mageagent:tools", "Hermes-3 8B Q8 (9GB)"),
            ("mageagent:validator", "Qwen-Coder 7B (5GB)"),
            ("mageagent:competitor", "Qwen-Coder 32B (18GB)")
        ]

        // Warm up each model sequentially
        warmupModels(modelsToWarmup, index: 0)
    }

    // MARK: - Model Loading Actions

    @objc func loadModelAction(_ sender: NSMenuItem) {
        guard let modelId = sender.representedObject as? String else {
            debugLog("ERROR: loadModelAction - no modelId in representedObject")
            return
        }

        // Find model info
        let modelName = availableModels.first { $0.modelId == modelId }?.displayName ?? modelId
        debugLog("Load Model action triggered for: \(modelName)")

        // Show loading notification
        sendNotification(title: "MageAgent", body: "Loading \(modelName)...")

        // Update status to show loading
        DispatchQueue.main.async {
            if let statusItem = self.menu.item(withTag: MenuItemTag.status.rawValue) {
                statusItem.title = "Loading: \(modelName)..."
            }
        }

        // Load the model via warmup request
        warmupModel(modelId: modelId) { [weak self] success in
            guard let self = self else { return }

            if success {
                self.loadedModels.insert(modelId)
                self.sendNotification(title: "MageAgent", body: "\(modelName) loaded successfully!")
                self.debugLog("Model \(modelName) loaded into memory")

                // Update menu item to show checkmark
                DispatchQueue.main.async {
                    sender.state = .on
                }
            } else {
                self.sendNotification(title: "MageAgent", body: "Failed to load \(modelName)")
                self.debugLog("Failed to load model \(modelName)")
            }

            // Restore status display
            self.checkServerStatus()
        }
    }

    // MARK: - Pattern Selection Actions

    @objc func selectPatternAction(_ sender: NSMenuItem) {
        guard let pattern = sender.representedObject as? PatternInfo else {
            debugLog("ERROR: selectPatternAction - no PatternInfo in representedObject")
            return
        }

        debugLog("Pattern selection triggered: \(pattern.displayName)")

        // Update selected pattern
        selectedPattern = pattern.patternId

        // Check which required models are not yet loaded
        let missingModels = pattern.requiredModels.filter { !loadedModels.contains($0) }

        if missingModels.isEmpty {
            // All models already loaded
            sendNotification(title: "MageAgent", body: "Pattern '\(pattern.displayName)' active - all required models already loaded!")
            updatePatternMenuCheckmarks(selectedPatternId: pattern.patternId)
        } else {
            // Need to load missing models
            let modelNames = missingModels.compactMap { modelId in
                availableModels.first { $0.modelId == modelId }?.displayName
            }.joined(separator: ", ")

            sendNotification(title: "MageAgent", body: "Loading models for '\(pattern.displayName)':\n\(modelNames)")

            // Load missing models sequentially
            loadModelsForPattern(pattern: pattern, modelIds: missingModels, index: 0)
        }
    }

    /// Load required models for a pattern sequentially
    private func loadModelsForPattern(pattern: PatternInfo, modelIds: [String], index: Int) {
        guard index < modelIds.count else {
            // All models loaded
            debugLog("All models for pattern '\(pattern.displayName)' loaded successfully")
            sendNotification(title: "MageAgent", body: "Pattern '\(pattern.displayName)' ready!")
            updatePatternMenuCheckmarks(selectedPatternId: pattern.patternId)
            checkServerStatus()
            return
        }

        let modelId = modelIds[index]
        let modelName = availableModels.first { $0.modelId == modelId }?.displayName ?? modelId

        // Update status
        DispatchQueue.main.async {
            if let statusItem = self.menu.item(withTag: MenuItemTag.status.rawValue) {
                statusItem.title = "Loading: \(modelName)..."
            }
        }

        debugLog("Loading model \(index + 1)/\(modelIds.count) for pattern: \(modelName)")

        warmupModel(modelId: modelId) { [weak self] success in
            guard let self = self else { return }

            if success {
                self.loadedModels.insert(modelId)
                self.debugLog("Model \(modelName) loaded for pattern")

                // Update checkmark on model menu item
                self.updateModelMenuCheckmark(modelId: modelId, loaded: true)
            } else {
                self.debugLog("Failed to load model \(modelName) for pattern")
            }

            // Continue to next model
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.loadModelsForPattern(pattern: pattern, modelIds: modelIds, index: index + 1)
            }
        }
    }

    /// Update checkmark on a model menu item
    private func updateModelMenuCheckmark(modelId: String, loaded: Bool) {
        if let modelsItem = menu.item(withTag: MenuItemTag.models.rawValue),
           let submenu = modelsItem.submenu {
            for item in submenu.items {
                if let itemModelId = item.representedObject as? String, itemModelId == modelId {
                    item.state = loaded ? .on : .off
                    break
                }
            }
        }
    }

    /// Update checkmarks on pattern menu items
    private func updatePatternMenuCheckmarks(selectedPatternId: String) {
        if let patternsItem = findPatternsMenuItem(),
           let submenu = patternsItem.submenu {
            for item in submenu.items {
                // Each pattern item has a submenu with details
                if let patternSubmenu = item.submenu {
                    // Look for "Use X Pattern" item
                    for subItem in patternSubmenu.items {
                        if let pattern = subItem.representedObject as? PatternInfo {
                            subItem.state = (pattern.patternId == selectedPatternId) ? .on : .off
                        }
                    }
                }
                // Also check the parent pattern item
                let patternId = availablePatterns.first { $0.displayName == item.title }?.patternId
                item.state = (patternId == selectedPatternId) ? .on : .off
            }
        }
    }

    /// Find the Patterns menu item by searching menu items
    private func findPatternsMenuItem() -> NSMenuItem? {
        for item in menu.items {
            if item.title == "Patterns" {
                return item
            }
        }
        return nil
    }

    private func warmupModels(_ models: [(String, String)], index: Int) {
        guard index < models.count else {
            // All models warmed up
            debugLog("All models warmed up successfully")
            sendNotification(title: "MageAgent", body: "All models loaded into memory!")
            return
        }

        let (modelId, modelName) = models[index]
        debugLog("Warming up model \(index + 1)/\(models.count): \(modelName)")

        // Update status to show progress
        DispatchQueue.main.async {
            if let statusItem = self.menu.item(withTag: MenuItemTag.status.rawValue) {
                statusItem.title = "Loading: \(modelName)..."
            }
        }

        // Send a minimal inference request to load the model
        warmupModel(modelId: modelId) { [weak self] success in
            guard let self = self else { return }

            if success {
                self.debugLog("Model \(modelName) loaded successfully")
            } else {
                self.debugLog("Failed to load model \(modelName)")
            }

            // Continue to next model after a brief delay
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.warmupModels(models, index: index + 1)
            }
        }
    }

    private func warmupModel(modelId: String, completion: @escaping (Bool) -> Void) {
        guard let url = URL(string: "\(Config.mageagentURL)/v1/chat/completions") else {
            completion(false)
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 300  // 5 minutes for large models

        // Minimal request to trigger model loading
        let body: [String: Any] = [
            "model": modelId,
            "messages": [
                ["role": "user", "content": "Hi"]
            ],
            "max_tokens": 5,
            "temperature": 0.1
        ]

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            debugLog("Failed to create warmup request: \(error)")
            completion(false)
            return
        }

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self?.debugLog("Warmup request failed: \(error.localizedDescription)")
                    completion(false)
                    return
                }

                guard let httpResponse = response as? HTTPURLResponse else {
                    completion(false)
                    return
                }

                self?.debugLog("Warmup response status: \(httpResponse.statusCode)")
                completion(httpResponse.statusCode == 200)

                // Restore status display
                self?.checkServerStatus()
            }
        }.resume()
    }

    private func executeServerCommand(_ command: String, successMessage: String, failureMessage: String) {
        runServerScript(command) { [weak self] success, output in
            guard let self = self else { return }

            if success {
                self.sendNotification(title: "MageAgent", body: successMessage)
                // Delay status check to allow server to start/stop
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    self.checkServerStatus()
                }
            } else {
                let errorDetail = output.isEmpty ? "" : "\n\(output.prefix(200))"
                self.sendNotification(title: "MageAgent", body: "\(failureMessage)\(errorDetail)")
            }
        }
    }

    // MARK: - Utility Actions

    @objc func openDocsAction(_ sender: NSMenuItem) {
        debugLog("Open API Docs action triggered")
        guard let url = URL(string: "\(Config.mageagentURL)/docs") else {
            debugLog("ERROR: Invalid docs URL")
            return
        }
        NSWorkspace.shared.open(url)
    }

    @objc func viewLogsAction(_ sender: NSMenuItem) {
        debugLog("View Logs action triggered")

        let logURL = URL(fileURLWithPath: Config.logFile)

        // Try to open with Console.app first
        if FileManager.default.fileExists(atPath: Config.logFile) {
            NSWorkspace.shared.open(logURL)
        } else {
            // Create empty log file if it doesn't exist
            let logDir = (Config.logFile as NSString).deletingLastPathComponent
            try? FileManager.default.createDirectory(atPath: logDir, withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: Config.logFile, contents: nil)
            NSWorkspace.shared.open(logURL)
        }
    }

    @objc func runTestAction(_ sender: NSMenuItem) {
        debugLog("Run Test action triggered")
        showTestResultsWindow()
    }

    // MARK: - Test Results Window

    /// Reference to keep test window alive
    private var testResultsWindow: NSWindow?
    private var testResultsTextView: NSTextView?

    /// Show streaming test results in a window
    private func showTestResultsWindow() {
        // Create window
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 600, height: 450),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "MageAgent Test Results"
        window.center()
        window.isReleasedWhenClosed = false

        // Create scroll view with text view
        let scrollView = NSScrollView(frame: NSRect(x: 0, y: 50, width: 600, height: 400))
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autoresizingMask = [.width, .height]
        scrollView.borderType = .noBorder

        let textView = NSTextView(frame: scrollView.bounds)
        textView.isEditable = false
        textView.isSelectable = true
        textView.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
        textView.backgroundColor = NSColor.textBackgroundColor
        textView.textColor = NSColor.textColor
        textView.autoresizingMask = [.width]
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.containerSize = NSSize(width: scrollView.contentSize.width, height: CGFloat.greatestFiniteMagnitude)

        scrollView.documentView = textView
        window.contentView?.addSubview(scrollView)

        // Create bottom bar with status and close button
        let bottomBar = NSView(frame: NSRect(x: 0, y: 0, width: 600, height: 50))
        bottomBar.autoresizingMask = [.width, .maxYMargin]

        let statusLabel = NSTextField(labelWithString: "Running tests...")
        statusLabel.frame = NSRect(x: 15, y: 15, width: 400, height: 20)
        statusLabel.font = NSFont.systemFont(ofSize: 13, weight: .medium)
        statusLabel.tag = 9999  // Tag to find later
        bottomBar.addSubview(statusLabel)

        let closeButton = NSButton(title: "Close", target: self, action: #selector(closeTestWindow(_:)))
        closeButton.frame = NSRect(x: 500, y: 10, width: 80, height: 30)
        closeButton.bezelStyle = .rounded
        closeButton.autoresizingMask = [.minXMargin]
        bottomBar.addSubview(closeButton)

        window.contentView?.addSubview(bottomBar)

        // Store references
        testResultsWindow = window
        testResultsTextView = textView

        // Show window
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        // Start running tests
        runStreamingTests()
    }

    @objc private func closeTestWindow(_ sender: Any) {
        testResultsWindow?.close()
        testResultsWindow = nil
        testResultsTextView = nil
    }

    /// Append text to test results with optional color
    private func appendTestResult(_ text: String, color: NSColor = .textColor) {
        guard let textView = testResultsTextView else { return }

        DispatchQueue.main.async {
            let attributedString = NSAttributedString(
                string: text,
                attributes: [
                    .font: NSFont.monospacedSystemFont(ofSize: 12, weight: .regular),
                    .foregroundColor: color
                ]
            )
            textView.textStorage?.append(attributedString)
            textView.scrollToEndOfDocument(nil)
        }
    }

    /// Update status label in test window
    private func updateTestStatus(_ status: String, color: NSColor = .labelColor) {
        DispatchQueue.main.async {
            if let bottomBar = self.testResultsWindow?.contentView?.subviews.first(where: { $0.frame.height == 50 }),
               let statusLabel = bottomBar.viewWithTag(9999) as? NSTextField {
                statusLabel.stringValue = status
                statusLabel.textColor = color
            }
        }
    }

    /// Run tests with streaming output
    private func runStreamingTests() {
        appendTestResult("MageAgent Test Suite\n", color: .systemBlue)
        appendTestResult("=" .padding(toLength: 50, withPad: "=", startingAt: 0) + "\n\n")

        // Define test cases
        let tests: [(name: String, test: (@escaping (Bool, String) -> Void) -> Void)] = [
            ("Server Health Check", testServerHealth),
            ("Models Endpoint", testModelsEndpoint),
            ("Validator Model (7B)", { self.testModel("mageagent:validator", completion: $0) }),
            ("Tools Model (Hermes-3)", { self.testModel("mageagent:tools", completion: $0) }),
            ("Primary Model (72B)", { self.testModel("mageagent:primary", completion: $0) }),
            ("Competitor Model (32B)", { self.testModel("mageagent:competitor", completion: $0) })
        ]

        runTestSequence(tests: tests, index: 0, passed: 0, failed: 0)
    }

    /// Run tests sequentially
    private func runTestSequence(tests: [(name: String, test: (@escaping (Bool, String) -> Void) -> Void)], index: Int, passed: Int, failed: Int) {
        guard index < tests.count else {
            // All tests complete
            appendTestResult("\n" + "=".padding(toLength: 50, withPad: "=", startingAt: 0) + "\n")

            let total = passed + failed
            let summary = "Results: \(passed)/\(total) passed"

            if failed == 0 {
                appendTestResult("All tests passed!\n", color: .systemGreen)
                updateTestStatus(summary, color: .systemGreen)
                sendNotification(title: "MageAgent Tests", body: "All \(total) tests passed!")
            } else {
                appendTestResult("\(failed) test(s) failed\n", color: .systemRed)
                updateTestStatus(summary, color: .systemRed)
                sendNotification(title: "MageAgent Tests", body: "\(failed) of \(total) tests failed")
            }
            return
        }

        let (name, testFunc) = tests[index]
        appendTestResult("[\(index + 1)/\(tests.count)] Testing: \(name)... ")
        updateTestStatus("Running: \(name)...")

        testFunc { [weak self] success, message in
            guard let self = self else { return }

            if success {
                self.appendTestResult("PASS\n", color: .systemGreen)
                if !message.isEmpty {
                    self.appendTestResult("    \(message)\n", color: .secondaryLabelColor)
                }
                self.runTestSequence(tests: tests, index: index + 1, passed: passed + 1, failed: failed)
            } else {
                self.appendTestResult("FAIL\n", color: .systemRed)
                if !message.isEmpty {
                    self.appendTestResult("    Error: \(message)\n", color: .systemRed)
                }
                self.runTestSequence(tests: tests, index: index + 1, passed: passed, failed: failed + 1)
            }
        }
    }

    // MARK: - Individual Test Functions

    private func testServerHealth(completion: @escaping (Bool, String) -> Void) {
        guard let url = URL(string: "\(Config.mageagentURL)/health") else {
            completion(false, "Invalid URL")
            return
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    completion(false, error.localizedDescription)
                    return
                }

                guard let httpResponse = response as? HTTPURLResponse else {
                    completion(false, "No HTTP response")
                    return
                }

                if httpResponse.statusCode == 200 {
                    if let data = data,
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let status = json["status"] as? String {
                        completion(true, "Status: \(status)")
                    } else {
                        completion(true, "")
                    }
                } else {
                    completion(false, "HTTP \(httpResponse.statusCode)")
                }
            }
        }.resume()
    }

    private func testModelsEndpoint(completion: @escaping (Bool, String) -> Void) {
        guard let url = URL(string: "\(Config.mageagentURL)/v1/models") else {
            completion(false, "Invalid URL")
            return
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 5.0

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    completion(false, error.localizedDescription)
                    return
                }

                guard let httpResponse = response as? HTTPURLResponse else {
                    completion(false, "No HTTP response")
                    return
                }

                if httpResponse.statusCode == 200 {
                    if let data = data,
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let models = json["data"] as? [[String: Any]] {
                        completion(true, "\(models.count) models available")
                    } else {
                        completion(true, "")
                    }
                } else {
                    completion(false, "HTTP \(httpResponse.statusCode)")
                }
            }
        }.resume()
    }

    private func testModel(_ modelId: String, completion: @escaping (Bool, String) -> Void) {
        guard let url = URL(string: "\(Config.mageagentURL)/v1/chat/completions") else {
            completion(false, "Invalid URL")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 120  // 2 minutes for model loading

        let body: [String: Any] = [
            "model": modelId,
            "messages": [["role": "user", "content": "Say 'test passed' in exactly 2 words."]],
            "max_tokens": 10,
            "temperature": 0.1
        ]

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            completion(false, "Failed to create request")
            return
        }

        let startTime = Date()

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                let elapsed = Date().timeIntervalSince(startTime)
                let timeStr = String(format: "%.1fs", elapsed)

                if let error = error {
                    completion(false, "\(error.localizedDescription) (\(timeStr))")
                    return
                }

                guard let httpResponse = response as? HTTPURLResponse else {
                    completion(false, "No HTTP response (\(timeStr))")
                    return
                }

                if httpResponse.statusCode == 200 {
                    if let data = data,
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let choices = json["choices"] as? [[String: Any]],
                       let firstChoice = choices.first,
                       let message = firstChoice["message"] as? [String: Any],
                       let content = message["content"] as? String {
                        let preview = content.prefix(30).replacingOccurrences(of: "\n", with: " ")
                        completion(true, "\"\(preview)\" (\(timeStr))")
                    } else {
                        completion(true, "Response received (\(timeStr))")
                    }
                } else {
                    var errorMsg = "HTTP \(httpResponse.statusCode)"
                    if let data = data,
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let detail = json["detail"] as? String {
                        errorMsg += ": \(detail)"
                    }
                    completion(false, "\(errorMsg) (\(timeStr))")
                }
            }
        }.resume()
    }

    @objc func showSettingsAction(_ sender: NSMenuItem) {
        debugLog("Settings action triggered")

        // Bring app to front for modal dialog
        NSApp.activate(ignoringOtherApps: true)

        let alert = NSAlert()
        alert.messageText = "MageAgent Settings"
        alert.informativeText = """
        MageAgent Server Configuration

        Port: 3457
        API: \(Config.mageagentURL)
        Script: \(Config.mageagentScript)
        Logs: \(Config.logFile)

        Loaded Models:
        • Primary: Qwen-72B Q8 (reasoning)
        • Tools: Hermes-3 8B Q8 (tool calling)
        • Validator: Qwen-Coder 7B (fast)
        • Competitor: Qwen-Coder 32B (coding)

        Available Patterns:
        • auto: Intelligent task routing
        • execute: Real tool execution (ReAct)
        • hybrid: Reasoning + tool extraction
        • validated: Generate + validate
        • compete: Multi-model with judge

        Status: \(isServerRunning ? "Running" : "Stopped")
        """
        alert.alertStyle = .informational
        alert.addButton(withTitle: "Close")
        alert.addButton(withTitle: "Open Logs Folder")

        let response = alert.runModal()
        if response == .alertSecondButtonReturn {
            let logsFolder = (Config.logFile as NSString).deletingLastPathComponent
            NSWorkspace.shared.open(URL(fileURLWithPath: logsFolder))
        }
    }

    // MARK: - Server Status Checking

    private func startStatusTimer() {
        statusTimer?.invalidate()
        statusTimer = Timer.scheduledTimer(withTimeInterval: Config.statusCheckInterval, repeats: true) { [weak self] _ in
            self?.checkServerStatus()
        }
        // Add to common run loop mode to ensure it fires even during menu tracking
        RunLoop.current.add(statusTimer!, forMode: .common)
    }

    private func checkServerStatus() {
        guard let url = URL(string: "\(Config.mageagentURL)/health") else {
            debugLog("ERROR: Invalid health check URL")
            return
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = Config.requestTimeout
        request.httpMethod = "GET"

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                guard let self = self else { return }

                if let error = error {
                    self.debugLog("Health check failed: \(error.localizedDescription)")
                    self.updateServerStatus(running: false, version: nil, models: [])
                    return
                }

                guard let httpResponse = response as? HTTPURLResponse,
                      httpResponse.statusCode == 200,
                      let data = data else {
                    self.updateServerStatus(running: false, version: nil, models: [])
                    return
                }

                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        let version = json["version"] as? String
                        let models = json["loaded_models"] as? [String] ?? []
                        self.updateServerStatus(running: true, version: version, models: models)
                    }
                } catch {
                    self.debugLog("Failed to parse health response: \(error)")
                    self.updateServerStatus(running: false, version: nil, models: [])
                }
            }
        }.resume()
    }

    private func updateServerStatus(running: Bool, version: String?, models: [String]) {
        isServerRunning = running

        // Update status menu item
        if let statusItem = menu.item(withTag: MenuItemTag.status.rawValue) {
            if running {
                let versionStr = version ?? "?"
                statusItem.title = "Status: Running (v\(versionStr))"
            } else {
                statusItem.title = "Status: Stopped"
            }
        }

        // Update loaded models set from server response
        loadedModels = Set(models.map { modelName in
            // Convert model names to our modelId format
            if modelName.contains("72B") || modelName.contains("primary") {
                return "mageagent:primary"
            } else if modelName.contains("Hermes") || modelName.contains("tools") {
                return "mageagent:tools"
            } else if modelName.contains("7B") || modelName.contains("validator") {
                return "mageagent:validator"
            } else if modelName.contains("32B") || modelName.contains("competitor") {
                return "mageagent:competitor"
            }
            return modelName
        })

        // Update checkmarks on model menu items based on loaded status
        if let modelsItem = menu.item(withTag: MenuItemTag.models.rawValue),
           let submenu = modelsItem.submenu {
            for item in submenu.items {
                if let modelId = item.representedObject as? String {
                    // Check if this model is loaded
                    item.state = loadedModels.contains(modelId) ? .on : .off
                }
            }
        }
    }

    // MARK: - Script Execution

    private func runServerScript(_ command: String, completion: @escaping (Bool, String) -> Void) {
        debugLog("Executing server script: \(command)")

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else {
                DispatchQueue.main.async { completion(false, "AppDelegate deallocated") }
                return
            }

            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/bin/bash")
            task.arguments = ["-l", "-c", "\(Config.mageagentScript) \(command)"]

            // Configure environment
            var environment = ProcessInfo.processInfo.environment
            environment["HOME"] = NSHomeDirectory()
            environment["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
            // Ensure Python/UV paths are available
            if let pyenvRoot = environment["PYENV_ROOT"] {
                environment["PATH"] = "\(pyenvRoot)/shims:\(environment["PATH"] ?? "")"
            }
            task.environment = environment

            let outputPipe = Pipe()
            let errorPipe = Pipe()
            task.standardOutput = outputPipe
            task.standardError = errorPipe

            do {
                try task.run()
                task.waitUntilExit()

                let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
                let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
                let output = String(data: outputData, encoding: .utf8) ?? ""
                let errorOutput = String(data: errorData, encoding: .utf8) ?? ""

                let combinedOutput = output + (errorOutput.isEmpty ? "" : "\nErrors: \(errorOutput)")

                self.debugLog("Script '\(command)' completed with status: \(task.terminationStatus)")
                self.debugLog("Output: \(combinedOutput.prefix(500))")

                DispatchQueue.main.async {
                    completion(task.terminationStatus == 0, combinedOutput)
                }
            } catch {
                self.debugLog("Script execution failed: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(false, error.localizedDescription)
                }
            }
        }
    }

    // MARK: - Notifications

    /// Track if notifications are authorized
    private var notificationsAuthorized: Bool = false

    private func requestNotificationPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { [weak self] granted, error in
            DispatchQueue.main.async {
                self?.notificationsAuthorized = granted
                if let error = error {
                    self?.debugLog("Notification permission error: \(error.localizedDescription)")
                } else {
                    self?.debugLog("Notification permission granted: \(granted)")
                }
            }
        }

        // Also check current authorization status
        UNUserNotificationCenter.current().getNotificationSettings { [weak self] settings in
            DispatchQueue.main.async {
                self?.notificationsAuthorized = settings.authorizationStatus == .authorized
                self?.debugLog("Notification settings: \(settings.authorizationStatus.rawValue)")
            }
        }
    }

    private func sendNotification(title: String, body: String) {
        debugLog("Sending notification: \(title) - \(body)")

        // Try modern UNUserNotificationCenter first
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil  // Deliver immediately
        )

        UNUserNotificationCenter.current().add(request) { [weak self] error in
            if let error = error {
                self?.debugLog("UNUserNotification failed: \(error.localizedDescription)")
                // Fallback to visual toast
                DispatchQueue.main.async {
                    self?.showToastAlert(title: title, body: body)
                }
            } else {
                self?.debugLog("Notification sent successfully")
            }
        }
    }

    /// Fallback toast using a floating panel
    private func showToastAlert(title: String, body: String) {
        // Create a small floating window as a toast
        let toastWindow = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 300, height: 80),
            styleMask: [.titled, .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )

        toastWindow.isFloatingPanel = true
        toastWindow.level = .floating
        toastWindow.titleVisibility = .hidden
        toastWindow.titlebarAppearsTransparent = true
        toastWindow.isMovableByWindowBackground = true
        toastWindow.backgroundColor = NSColor.windowBackgroundColor.withAlphaComponent(0.95)

        // Create content view
        let contentView = NSView(frame: NSRect(x: 0, y: 0, width: 300, height: 80))

        // Icon
        let iconView = NSImageView(frame: NSRect(x: 15, y: 22, width: 36, height: 36))
        if let iconImage = NSImage(contentsOfFile: Config.iconPath) {
            iconImage.size = NSSize(width: 36, height: 36)
            iconView.image = iconImage
        } else if let symbolImage = NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "MageAgent") {
            iconView.image = symbolImage
        }
        contentView.addSubview(iconView)

        // Title
        let titleLabel = NSTextField(labelWithString: title)
        titleLabel.frame = NSRect(x: 60, y: 48, width: 225, height: 20)
        titleLabel.font = NSFont.boldSystemFont(ofSize: 13)
        titleLabel.textColor = NSColor.labelColor
        contentView.addSubview(titleLabel)

        // Body
        let bodyLabel = NSTextField(labelWithString: body)
        bodyLabel.frame = NSRect(x: 60, y: 12, width: 225, height: 32)
        bodyLabel.font = NSFont.systemFont(ofSize: 11)
        bodyLabel.textColor = NSColor.secondaryLabelColor
        bodyLabel.lineBreakMode = .byTruncatingTail
        bodyLabel.maximumNumberOfLines = 2
        contentView.addSubview(bodyLabel)

        toastWindow.contentView = contentView

        // Position in top-right corner
        if let screen = NSScreen.main {
            let screenFrame = screen.visibleFrame
            let windowFrame = toastWindow.frame
            let x = screenFrame.maxX - windowFrame.width - 20
            let y = screenFrame.maxY - windowFrame.height - 10
            toastWindow.setFrameOrigin(NSPoint(x: x, y: y))
        }

        // Show with animation
        toastWindow.alphaValue = 0
        toastWindow.makeKeyAndOrderFront(nil)
        toastWindow.orderFrontRegardless()

        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.3
            toastWindow.animator().alphaValue = 1.0
        }

        // Auto-dismiss after 3 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
            NSAnimationContext.runAnimationGroup({ context in
                context.duration = 0.3
                toastWindow.animator().alphaValue = 0
            }) {
                toastWindow.close()
            }
        }
    }

    // MARK: - System Pressure Monitoring

    /// Start the pressure monitoring timer
    private func startPressureTimer() {
        pressureTimer?.invalidate()
        pressureTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.updateSystemPressure()
        }
        // Add to common run loop mode
        RunLoop.current.add(pressureTimer!, forMode: .common)

        // Initial update
        updateSystemPressure()
    }

    /// Update all system pressure indicators
    private func updateSystemPressure() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self = self else { return }

            // Get memory info
            let memInfo = self.getMemoryInfo()

            // Get CPU usage
            let cpuUsage = self.getCPUUsage()

            // Get GPU info via Metal/powermetrics
            let gpuInfo = self.getGPUInfo()

            DispatchQueue.main.async {
                self.updatePressureMenuItems(memory: memInfo, cpu: cpuUsage, gpu: gpuInfo)
            }
        }
    }

    /// Get memory information using vm_statistics64
    private func getMemoryInfo() -> (usedGB: Double, totalGB: Double, pressure: SystemPressure, percentUsed: Double) {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)

        let hostPort = mach_host_self()
        let result = withUnsafeMutablePointer(to: &stats) { statsPtr in
            statsPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                host_statistics64(hostPort, HOST_VM_INFO64, intPtr, &count)
            }
        }

        guard result == KERN_SUCCESS else {
            return (0, 0, .nominal, 0)
        }

        let pageSize = UInt64(vm_kernel_page_size)

        // Calculate memory usage
        let active = UInt64(stats.active_count) * pageSize
        let wired = UInt64(stats.wire_count) * pageSize
        let compressed = UInt64(stats.compressor_page_count) * pageSize

        // Used = Active + Wired + Compressed (similar to Activity Monitor "Memory Used")
        let used = active + wired + compressed

        // Get total physical memory
        var totalMemory: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        sysctlbyname("hw.memsize", &totalMemory, &size, nil, 0)

        let usedGB = Double(used) / (1024 * 1024 * 1024)
        let totalGB = Double(totalMemory) / (1024 * 1024 * 1024)
        let percentUsed = (Double(used) / Double(totalMemory)) * 100

        // Determine pressure level based on percentage
        // Apple Silicon typically shows warning around 80%, critical around 90%
        let pressure: SystemPressure
        if percentUsed >= 90 {
            pressure = .critical
        } else if percentUsed >= 75 {
            pressure = .warning
        } else {
            pressure = .nominal
        }

        return (usedGB, totalGB, pressure, percentUsed)
    }

    /// Get CPU usage using host_processor_info
    private func getCPUUsage() -> (usage: Double, pressure: SystemPressure) {
        var numCPUs: natural_t = 0
        var cpuInfo: processor_info_array_t?
        var numCPUInfo: mach_msg_type_number_t = 0

        let result = host_processor_info(mach_host_self(),
                                         PROCESSOR_CPU_LOAD_INFO,
                                         &numCPUs,
                                         &cpuInfo,
                                         &numCPUInfo)

        guard result == KERN_SUCCESS, let cpuInfo = cpuInfo else {
            return (0, .nominal)
        }

        var totalUser: UInt64 = 0
        var totalSystem: UInt64 = 0
        var totalIdle: UInt64 = 0
        var totalNice: UInt64 = 0

        for i in 0..<Int32(numCPUs) {
            let offset = Int(i) * Int(CPU_STATE_MAX)
            totalUser += UInt64(cpuInfo[offset + Int(CPU_STATE_USER)])
            totalSystem += UInt64(cpuInfo[offset + Int(CPU_STATE_SYSTEM)])
            totalIdle += UInt64(cpuInfo[offset + Int(CPU_STATE_IDLE)])
            totalNice += UInt64(cpuInfo[offset + Int(CPU_STATE_NICE)])
        }

        // Calculate delta from previous
        let userDelta = totalUser - previousCPUTicks.user
        let systemDelta = totalSystem - previousCPUTicks.system
        let idleDelta = totalIdle - previousCPUTicks.idle
        let niceDelta = totalNice - previousCPUTicks.nice

        // Store current values for next calculation
        previousCPUTicks = (totalUser, totalSystem, totalIdle, totalNice)

        let totalDelta = userDelta + systemDelta + idleDelta + niceDelta
        guard totalDelta > 0 else {
            return (0, .nominal)
        }

        let usage = Double(userDelta + systemDelta + niceDelta) / Double(totalDelta) * 100

        // Deallocate
        let cpuInfoSize = vm_size_t(numCPUInfo) * vm_size_t(MemoryLayout<integer_t>.size)
        vm_deallocate(mach_task_self_, vm_address_t(bitPattern: cpuInfo), cpuInfoSize)

        // Determine pressure
        let pressure: SystemPressure
        if usage >= 90 {
            pressure = .critical
        } else if usage >= 70 {
            pressure = .warning
        } else {
            pressure = .nominal
        }

        return (usage, pressure)
    }

    /// Get GPU/Metal information
    private func getGPUInfo() -> (description: String, pressure: SystemPressure) {
        // For Apple Silicon, GPU is part of the unified memory
        // We can estimate GPU pressure based on memory pressure when models are loaded

        // Check if any large models are loaded (primary = 72B uses ~77GB)
        let hasLargeModelLoaded = loadedModels.contains("mageagent:primary")

        // Get memory pressure to infer GPU pressure (unified memory)
        let memInfo = getMemoryInfo()

        // If large model is loaded and memory pressure is high, GPU is likely under pressure
        var pressure: SystemPressure = .nominal
        var description = "Metal: Idle"

        if isServerRunning {
            if hasLargeModelLoaded {
                description = "Metal: Active (72B loaded)"
                if memInfo.pressure == .critical {
                    pressure = .critical
                    description = "Metal: Heavy Load (72B)"
                } else if memInfo.pressure == .warning {
                    pressure = .warning
                    description = "Metal: Moderate (72B)"
                }
            } else if !loadedModels.isEmpty {
                let modelCount = loadedModels.count
                description = "Metal: Active (\(modelCount) model\(modelCount > 1 ? "s" : ""))"
            } else {
                description = "Metal: Standby"
            }
        }

        return (description, pressure)
    }

    /// Update the menu items with current pressure information
    private func updatePressureMenuItems(memory: (usedGB: Double, totalGB: Double, pressure: SystemPressure, percentUsed: Double),
                                         cpu: (usage: Double, pressure: SystemPressure),
                                         gpu: (description: String, pressure: SystemPressure)) {
        // Determine overall system pressure (worst of all)
        let overallPressure: SystemPressure
        if memory.pressure == .critical || cpu.pressure == .critical || gpu.pressure == .critical {
            overallPressure = .critical
        } else if memory.pressure == .warning || cpu.pressure == .warning || gpu.pressure == .warning {
            overallPressure = .warning
        } else {
            overallPressure = .nominal
        }

        // CRITICAL: Use explicit black/white based on appearance, NOT semantic colors
        // macOS dims all semantic colors for disabled menu items, so we must use raw colors
        let isDarkMode = NSApp.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        let textColor = isDarkMode ? NSColor.white : NSColor.black

        // Update header with overall pressure
        if let headerItem = menu.item(withTag: MenuItemTag.systemPressure.rawValue) {
            let headerAttr = NSMutableAttributedString(string: "\(overallPressure.indicator) ", attributes: [.foregroundColor: overallPressure.color])
            headerAttr.append(NSAttributedString(string: "System Resources", attributes: [
                .foregroundColor: textColor,
                .font: NSFont.systemFont(ofSize: 13, weight: .bold)
            ]))
            headerItem.attributedTitle = headerAttr
        }

        // Update memory detail - use explicit black/white for full opacity
        if let memItem = menu.item(withTag: MenuItemTag.memoryDetail.rawValue) {
            let memText = String(format: "Memory: %.1f / %.1f GB (%.0f%%)", memory.usedGB, memory.totalGB, memory.percentUsed)
            let memAttr = NSMutableAttributedString(string: "  \(memory.pressure.indicator) ", attributes: [.foregroundColor: memory.pressure.color])
            memAttr.append(NSAttributedString(string: memText, attributes: [
                .foregroundColor: textColor,
                .font: NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .semibold)
            ]))
            memItem.attributedTitle = memAttr
        }

        // Update CPU detail - use explicit black/white
        if let cpuItem = menu.item(withTag: MenuItemTag.cpuDetail.rawValue) {
            let cpuText = String(format: "CPU: %.1f%%", cpu.usage)
            let cpuAttr = NSMutableAttributedString(string: "  \(cpu.pressure.indicator) ", attributes: [.foregroundColor: cpu.pressure.color])
            cpuAttr.append(NSAttributedString(string: cpuText, attributes: [
                .foregroundColor: textColor,
                .font: NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .semibold)
            ]))
            cpuItem.attributedTitle = cpuAttr
        }

        // Update GPU detail - use explicit black/white
        if let gpuItem = menu.item(withTag: MenuItemTag.gpuDetail.rawValue) {
            let gpuAttr = NSMutableAttributedString(string: "  \(gpu.pressure.indicator) ", attributes: [.foregroundColor: gpu.pressure.color])
            gpuAttr.append(NSAttributedString(string: gpu.description, attributes: [
                .foregroundColor: textColor,
                .font: NSFont.systemFont(ofSize: 12, weight: .semibold)
            ]))
            gpuItem.attributedTitle = gpuAttr
        }

        // Store current values
        memoryPressure = memory.pressure
        cpuUsage = cpu.usage
    }

    // MARK: - Debug Logging

    private func debugLog(_ message: String) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let logMessage = "[\(timestamp)] \(message)"

        // Print to console
        print(logMessage)

        // Write to debug log file
        let logEntry = logMessage + "\n"
        guard let data = logEntry.data(using: .utf8) else { return }

        let logDir = (Config.debugLogFile as NSString).deletingLastPathComponent

        // Ensure log directory exists
        if !FileManager.default.fileExists(atPath: logDir) {
            try? FileManager.default.createDirectory(atPath: logDir, withIntermediateDirectories: true)
        }

        if FileManager.default.fileExists(atPath: Config.debugLogFile) {
            if let fileHandle = FileHandle(forWritingAtPath: Config.debugLogFile) {
                defer { fileHandle.closeFile() }
                fileHandle.seekToEndOfFile()
                fileHandle.write(data)
            }
        } else {
            try? data.write(to: URL(fileURLWithPath: Config.debugLogFile))
        }
    }
}
