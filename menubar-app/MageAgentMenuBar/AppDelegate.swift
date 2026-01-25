import Cocoa
import UserNotifications
import IOKit

// MARK: - NSBezierPath Extension for CGPath conversion
extension NSBezierPath {
    var cgPath: CGPath {
        let path = CGMutablePath()
        var points = [CGPoint](repeating: .zero, count: 3)

        for i in 0..<elementCount {
            let type = element(at: i, associatedPoints: &points)
            switch type {
            case .moveTo:
                path.move(to: points[0])
            case .lineTo:
                path.addLine(to: points[0])
            case .curveTo:
                path.addCurve(to: points[2], control1: points[0], control2: points[1])
            case .closePath:
                path.closeSubpath()
            case .cubicCurveTo:
                path.addCurve(to: points[2], control1: points[0], control2: points[1])
            case .quadraticCurveTo:
                path.addQuadCurve(to: points[1], control: points[0])
            @unknown default:
                break
            }
        }
        return path
    }
}

// MARK: - Nexus Local Compute Application Delegate
// Production-grade macOS menu bar application for local LLM orchestration
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

    /// Server version from health endpoint
    private var serverVersion: String = "?"

    /// Offline mode status
    private var isOfflineMode: Bool = false

    /// Model router status
    private var isRouterRunning: Bool = false
    private var currentRouterProvider: String = "anthropic"
    private var currentRouterModel: String = "claude-sonnet-4-5-20250929"

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
        case neuralEngineDetail = 604
        case throughputDetail = 605
        case ragStatus = 650
        case offlineMode = 700
        case modelRouter = 800
        case routerStatus = 801
        case switchToOpus = 810
        case switchToSonnet = 811
        case switchToMageHybrid = 812
        case switchToMageExecute = 813
        case switchToMageFast = 814
        case chatInterface = 900
    }

    // MARK: - Configuration

    private struct Config {
        static let mageagentScript = "\(NSHomeDirectory())/.claude/scripts/mageagent-server.sh"
        static let offlineModeScript = "\(NSHomeDirectory())/.claude/scripts/offline-mode.sh"
        static let modelRouterScript = "\(NSHomeDirectory())/.claude/scripts/model-router.sh"
        static let routerStateFile = "\(NSHomeDirectory())/.claude/router-state.json"
        static let mageagentURL = "http://localhost:3457"
        static let routerURL = "http://localhost:3456"
        static let logFile = "\(NSHomeDirectory())/.claude/debug/mageagent.log"
        static let debugLogFile = "\(NSHomeDirectory())/.claude/debug/mageagent-menubar-debug.log"
        static let iconPath = "\(NSHomeDirectory())/.claude/mageagent-menubar/icons/icon_18x18@2x.png"
        static let adverantLogoPath = "\(NSHomeDirectory())/.claude/mageagent-menubar/icons/adverant-logo.png"
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
        ModelInfo(modelId: "mageagent:competitor", displayName: "Qwen-Coder 32B (18GB) - Coding", memorySize: "18GB"),
        ModelInfo(modelId: "mageagent:glm", displayName: "GLM-4.7 Flash (17GB) - Fast MoE", memorySize: "17GB")
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
        ),
        PatternInfo(
            patternId: "mageagent:self_consistent",
            displayName: "self_consistent (+17%)",
            requiredModels: ["mageagent:primary", "mageagent:validator"],
            description: "Multiple reasoning paths - generates 3 responses, selects most consistent"
        ),
        PatternInfo(
            patternId: "mageagent:critic",
            displayName: "critic (+10%)",
            requiredModels: ["mageagent:primary", "mageagent:validator"],
            description: "CRITIC framework - self-critique loop for edge case detection"
        )
    ]

    /// Track which models are currently loaded
    private var loadedModels: Set<String> = []

    /// Track which models are disabled (cannot be loaded or used)
    /// Persisted to UserDefaults
    private var disabledModels: Set<String> = []
    private let disabledModelsKey = "com.adverant.nexus.disabledModels"

    /// Model presets (persisted to UserDefaults)
    private struct ModelPreset: Codable {
        let name: String
        let models: [String]
        let isBuiltIn: Bool
    }

    private var customPresets: [ModelPreset] = []
    private let customPresetsKey = "com.adverant.nexus.customPresets"
    private var currentPresetName: String = "None"

    /// Built-in presets
    private let builtInPresets: [ModelPreset] = [
        ModelPreset(name: "Full Power", models: ["mageagent:primary", "mageagent:tools", "mageagent:validator", "mageagent:competitor"], isBuiltIn: true),
        ModelPreset(name: "Coding Mode", models: ["mageagent:competitor", "mageagent:tools"], isBuiltIn: true),
        ModelPreset(name: "Fast Mode", models: ["mageagent:validator", "mageagent:glm"], isBuiltIn: true),
        ModelPreset(name: "Hybrid Only", models: ["mageagent:primary", "mageagent:tools"], isBuiltIn: true),
        ModelPreset(name: "Tool Calling", models: ["mageagent:tools"], isBuiltIn: true)
    ]

    /// Currently selected pattern
    private var selectedPattern: String = "mageagent:auto"

    /// System memory info cache
    private var lastMemoryInfo: (used: UInt64, total: UInt64, pressure: String) = (0, 0, "nominal")

    /// Previous CPU ticks for delta calculation
    private var previousCPUTicks: (user: UInt64, system: UInt64, idle: UInt64, nice: UInt64) = (0, 0, 0, 0)

    /// Neural Engine TOPS (Tera Operations Per Second) - Apple Silicon specific
    /// M4 Max = 38 TOPS, M3 Max = 18 TOPS, M2 Max = 15.8 TOPS, M1 Max = 11 TOPS
    private var neuralEngineTOPS: Double = 0

    /// Token throughput tracking
    private var lastThroughput: (tokensPerSec: Double, totalTokens: Int, lastModel: String) = (0, 0, "")

    /// Model memory sizes in GB (for status display)
    private let modelSizes: [String: Double] = [
        "primary": 77.0,
        "mageagent:primary": 77.0,
        "competitor": 18.0,
        "mageagent:competitor": 18.0,
        "tools": 9.0,
        "mageagent:tools": 9.0,
        "validator": 5.0,
        "mageagent:validator": 5.0,
        "glm": 17.0,
        "mageagent:glm": 17.0
    ]

    /// Memory pressure history for graph (stores last 60 samples)
    private var memoryPressureHistory: [Double] = []
    private let maxHistorySamples = 60

    // MARK: - Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        debugLog("Application launched - initializing Nexus Local Compute")

        // Configure as menu bar only app (no dock icon)
        NSApp.setActivationPolicy(.accessory)

        // Load disabled models from UserDefaults
        loadDisabledModels()

        // Load custom presets from UserDefaults
        loadCustomPresets()

        // Detect Neural Engine TOPS for this Apple Silicon chip
        detectNeuralEngineTOPS()

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

        // Check initial offline mode status
        checkOfflineModeStatus()

        debugLog("Application initialization complete")
    }

    /// Detect Neural Engine TOPS based on Apple Silicon chip model
    private func detectNeuralEngineTOPS() {
        var size: Int = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var chipName = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &chipName, &size, nil, 0)
        let chipString = String(cString: chipName)

        // Neural Engine TOPS by chip generation
        // Reference: Apple Silicon specifications
        if chipString.contains("M4") {
            if chipString.contains("Max") {
                neuralEngineTOPS = 38.0  // M4 Max
            } else if chipString.contains("Pro") {
                neuralEngineTOPS = 38.0  // M4 Pro
            } else {
                neuralEngineTOPS = 38.0  // M4 base
            }
        } else if chipString.contains("M3") {
            if chipString.contains("Max") {
                neuralEngineTOPS = 18.0  // M3 Max
            } else if chipString.contains("Pro") {
                neuralEngineTOPS = 18.0  // M3 Pro
            } else {
                neuralEngineTOPS = 18.0  // M3 base
            }
        } else if chipString.contains("M2") {
            if chipString.contains("Max") {
                neuralEngineTOPS = 15.8  // M2 Max
            } else if chipString.contains("Pro") {
                neuralEngineTOPS = 15.8  // M2 Pro
            } else if chipString.contains("Ultra") {
                neuralEngineTOPS = 31.6  // M2 Ultra (2x M2 Max)
            } else {
                neuralEngineTOPS = 15.8  // M2 base
            }
        } else if chipString.contains("M1") {
            if chipString.contains("Max") {
                neuralEngineTOPS = 11.0  // M1 Max
            } else if chipString.contains("Pro") {
                neuralEngineTOPS = 11.0  // M1 Pro
            } else if chipString.contains("Ultra") {
                neuralEngineTOPS = 22.0  // M1 Ultra (2x M1 Max)
            } else {
                neuralEngineTOPS = 11.0  // M1 base
            }
        } else {
            neuralEngineTOPS = 0  // Unknown or Intel
        }

        debugLog("Detected chip: \(chipString), Neural Engine: \(neuralEngineTOPS) TOPS")
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
        } else if let symbolImage = NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "Nexus Local Compute") {
            symbolImage.isTemplate = true
            button.image = symbolImage
            debugLog("Using fallback system symbol")
        } else {
            // Ultimate fallback - text
            button.title = "NLC"
            debugLog("Using text fallback for status item")
        }

        button.toolTip = "Nexus Local Compute - Multi-Model AI Orchestration"
    }

    private func setupMenu() {
        menu = NSMenu()
        menu.autoenablesItems = false  // We manage enabled state manually

        // Pop Out button at the very top - opens draggable panel
        let popOutItem = NSMenuItem(title: "⬜ Pop Out Dashboard", action: #selector(popOutDashboardAction(_:)), keyEquivalent: "p")
        popOutItem.target = self
        popOutItem.keyEquivalentModifierMask = [.command, .shift]
        menu.addItem(popOutItem)

        menu.addItem(NSMenuItem.separator())

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

        // Neural Engine TOPS item - enabled with no-op action for full opacity
        let neItem = NSMenuItem(title: "  Neural Engine: Detecting...", action: #selector(displayOnlyAction(_:)), keyEquivalent: "")
        neItem.tag = MenuItemTag.neuralEngineDetail.rawValue
        neItem.target = self
        neItem.isEnabled = true
        menu.addItem(neItem)

        // Throughput item - enabled with no-op action for full opacity
        let throughputItem = NSMenuItem(title: "  Throughput: Idle", action: #selector(displayOnlyAction(_:)), keyEquivalent: "")
        throughputItem.tag = MenuItemTag.throughputDetail.rawValue
        throughputItem.target = self
        throughputItem.isEnabled = true
        menu.addItem(throughputItem)

        // Memory Pressure Graph - Activity Monitor style (after Tokens)
        let memPressureBarItem = NSMenuItem()
        memPressureBarItem.tag = 610  // Custom tag for pressure graph
        let barView = createMemoryPressureBarView()
        memPressureBarItem.view = barView
        menu.addItem(memPressureBarItem)

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

        // Model status summary header (will be updated dynamically)
        let statusSummaryItem = NSMenuItem(title: "Loaded: 0 models (0GB)", action: nil, keyEquivalent: "")
        statusSummaryItem.isEnabled = false
        statusSummaryItem.tag = 5000  // Special tag for status summary
        modelsSubmenu.addItem(statusSummaryItem)
        modelsSubmenu.addItem(NSMenuItem.separator())

        // Add header
        let headerItem = NSMenuItem(title: "Click to load/unload (✓ = loaded):", action: nil, keyEquivalent: "")
        headerItem.isEnabled = false
        modelsSubmenu.addItem(headerItem)

        // Add each model as a clickable item with status indicators
        for (index, model) in availableModels.enumerated() {
            let isLoaded = loadedModels.contains(model.modelId)
            let isDisabled = disabledModels.contains(model.modelId)

            var title: String
            if isDisabled {
                title = "⊘ \(model.displayName) [DISABLED]"
            } else if isLoaded {
                title = "✓ \(model.displayName)"
            } else {
                title = "○ \(model.displayName)"
            }

            let item = NSMenuItem(title: title, action: #selector(toggleModelLoadAction(_:)), keyEquivalent: "")
            item.target = self
            item.tag = 1000 + index  // Tags starting at 1000 for models
            item.representedObject = model.modelId
            item.state = isLoaded ? .on : .off
            item.isEnabled = !isDisabled

            if isDisabled {
                item.attributedTitle = NSAttributedString(
                    string: title,
                    attributes: [.foregroundColor: NSColor.disabledControlTextColor]
                )
            }

            modelsSubmenu.addItem(item)
        }

        modelsSubmenu.addItem(NSMenuItem.separator())

        modelsSubmenu.addItem(NSMenuItem.separator())

        // Load all models option
        let loadAllItem = NSMenuItem(title: "Load All Models", action: #selector(warmupModelsAction(_:)), keyEquivalent: "w")
        loadAllItem.target = self
        loadAllItem.keyEquivalentModifierMask = [.command]
        modelsSubmenu.addItem(loadAllItem)

        // Unload submenu
        let unloadMenuItem = NSMenuItem(title: "Unload Model...", action: nil, keyEquivalent: "")
        let unloadSubmenu = NSMenu()

        for (index, model) in availableModels.enumerated() {
            let item = NSMenuItem(title: model.displayName, action: #selector(unloadModelAction(_:)), keyEquivalent: "")
            item.target = self
            item.tag = 3000 + index  // Tags starting at 3000 for unload
            item.representedObject = model.modelId
            unloadSubmenu.addItem(item)
        }

        unloadSubmenu.addItem(NSMenuItem.separator())

        let unloadAllItem = NSMenuItem(title: "Unload ALL Models", action: #selector(unloadAllModelsAction(_:)), keyEquivalent: "u")
        unloadAllItem.target = self
        unloadAllItem.keyEquivalentModifierMask = [.command, .shift]
        unloadSubmenu.addItem(unloadAllItem)

        unloadMenuItem.submenu = unloadSubmenu
        modelsSubmenu.addItem(unloadMenuItem)

        // Disable/Enable submenu (persisted setting)
        let disableMenuItem = NSMenuItem(title: "Disable/Enable Model...", action: nil, keyEquivalent: "")
        let disableSubmenu = NSMenu()

        let disableHeader = NSMenuItem(title: "Disabled models cannot be loaded:", action: nil, keyEquivalent: "")
        disableHeader.isEnabled = false
        disableSubmenu.addItem(disableHeader)
        disableSubmenu.addItem(NSMenuItem.separator())

        for (index, model) in availableModels.enumerated() {
            let isDisabled = disabledModels.contains(model.modelId)
            let title = isDisabled ? "✓ Enable \(model.displayName)" : "⊘ Disable \(model.displayName)"
            let item = NSMenuItem(title: title, action: #selector(toggleModelDisabledAction(_:)), keyEquivalent: "")
            item.target = self
            item.tag = 4000 + index  // Tags starting at 4000 for disable toggle
            item.representedObject = model.modelId
            item.state = isDisabled ? .on : .off
            disableSubmenu.addItem(item)
        }

        disableMenuItem.submenu = disableSubmenu
        modelsSubmenu.addItem(disableMenuItem)

        modelsSubmenu.addItem(NSMenuItem.separator())

        // Presets submenu
        let presetsMenuItem = NSMenuItem(title: "Presets...", action: nil, keyEquivalent: "")
        let presetsSubmenu = NSMenu()

        // Built-in presets header
        let builtInHeader = NSMenuItem(title: "Built-in Presets:", action: nil, keyEquivalent: "")
        builtInHeader.isEnabled = false
        presetsSubmenu.addItem(builtInHeader)

        for preset in builtInPresets {
            let memoryTotal = preset.models.compactMap { modelId -> Double? in
                let key = modelId.replacingOccurrences(of: "mageagent:", with: "")
                return modelSizes[key] ?? modelSizes[modelId]
            }.reduce(0, +)

            let item = NSMenuItem(
                title: "\(preset.name) (\(Int(memoryTotal))GB)",
                action: #selector(applyPresetAction(_:)),
                keyEquivalent: ""
            )
            item.target = self
            item.representedObject = preset
            presetsSubmenu.addItem(item)
        }

        presetsSubmenu.addItem(NSMenuItem.separator())

        // Custom presets header
        let customHeader = NSMenuItem(title: "Custom Presets:", action: nil, keyEquivalent: "")
        customHeader.isEnabled = false
        presetsSubmenu.addItem(customHeader)

        if customPresets.isEmpty {
            let emptyItem = NSMenuItem(title: "  (none saved)", action: nil, keyEquivalent: "")
            emptyItem.isEnabled = false
            presetsSubmenu.addItem(emptyItem)
        } else {
            for preset in customPresets {
                let memoryTotal = preset.models.compactMap { modelId -> Double? in
                    let key = modelId.replacingOccurrences(of: "mageagent:", with: "")
                    return modelSizes[key] ?? modelSizes[modelId]
                }.reduce(0, +)

                let item = NSMenuItem(
                    title: "\(preset.name) (\(Int(memoryTotal))GB)",
                    action: #selector(applyPresetAction(_:)),
                    keyEquivalent: ""
                )
                item.target = self
                item.representedObject = preset
                presetsSubmenu.addItem(item)
            }
        }

        presetsSubmenu.addItem(NSMenuItem.separator())

        // Save current as preset
        let savePresetItem = NSMenuItem(
            title: "Save Current as Preset...",
            action: #selector(saveCurrentAsPresetAction(_:)),
            keyEquivalent: ""
        )
        savePresetItem.target = self
        presetsSubmenu.addItem(savePresetItem)

        presetsMenuItem.submenu = presetsSubmenu
        modelsSubmenu.addItem(presetsMenuItem)

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

        // RAG Status item (clickable to configure)
        let ragItem = NSMenuItem(title: "GraphRAG: Checking...", action: #selector(configureRAGAction(_:)), keyEquivalent: "")
        ragItem.tag = MenuItemTag.ragStatus.rawValue
        ragItem.target = self
        ragItem.isEnabled = true
        ragItem.toolTip = "Click to configure GraphRAG (Temporal Knowledge Graph) for a project"
        menu.addItem(ragItem)

        // Offline Mode Toggle
        let offlineItem = NSMenuItem(title: "Offline Mode", action: #selector(toggleOfflineModeAction(_:)), keyEquivalent: "o")
        offlineItem.target = self
        offlineItem.tag = MenuItemTag.offlineMode.rawValue
        offlineItem.keyEquivalentModifierMask = [.command, .shift]
        menu.addItem(offlineItem)

        menu.addItem(NSMenuItem.separator())

        // Model Router Section
        let routerItem = NSMenuItem(title: "Model Router", action: nil, keyEquivalent: "")
        routerItem.tag = MenuItemTag.modelRouter.rawValue
        let routerSubmenu = NSMenu()

        // Router status
        let routerStatusItem = NSMenuItem(title: "Router: Checking...", action: #selector(displayOnlyAction(_:)), keyEquivalent: "")
        routerStatusItem.tag = MenuItemTag.routerStatus.rawValue
        routerStatusItem.target = self
        routerStatusItem.isEnabled = true
        routerSubmenu.addItem(routerStatusItem)

        routerSubmenu.addItem(NSMenuItem.separator())

        // Start/Stop Router
        let startRouterItem = NSMenuItem(title: "Start Router", action: #selector(startRouterAction(_:)), keyEquivalent: "")
        startRouterItem.target = self
        routerSubmenu.addItem(startRouterItem)

        let stopRouterItem = NSMenuItem(title: "Stop Router", action: #selector(stopRouterAction(_:)), keyEquivalent: "")
        stopRouterItem.target = self
        routerSubmenu.addItem(stopRouterItem)

        routerSubmenu.addItem(NSMenuItem.separator())

        // Model switching options
        let modelHeader = NSMenuItem(title: "Switch Model:", action: nil, keyEquivalent: "")
        modelHeader.isEnabled = false
        routerSubmenu.addItem(modelHeader)

        let opusItem = NSMenuItem(title: "  Claude Opus 4.5", action: #selector(switchToOpusAction(_:)), keyEquivalent: "1")
        opusItem.tag = MenuItemTag.switchToOpus.rawValue
        opusItem.target = self
        opusItem.keyEquivalentModifierMask = [.command, .option]
        routerSubmenu.addItem(opusItem)

        let sonnetItem = NSMenuItem(title: "  Claude Sonnet 4.5", action: #selector(switchToSonnetAction(_:)), keyEquivalent: "2")
        sonnetItem.tag = MenuItemTag.switchToSonnet.rawValue
        sonnetItem.target = self
        sonnetItem.keyEquivalentModifierMask = [.command, .option]
        routerSubmenu.addItem(sonnetItem)

        routerSubmenu.addItem(NSMenuItem.separator())

        let mageHybridItem = NSMenuItem(title: "  Local AI: Hybrid (72B+8B)", action: #selector(switchToMageHybridAction(_:)), keyEquivalent: "3")
        mageHybridItem.tag = MenuItemTag.switchToMageHybrid.rawValue
        mageHybridItem.target = self
        mageHybridItem.keyEquivalentModifierMask = [.command, .option]
        routerSubmenu.addItem(mageHybridItem)

        let mageExecuteItem = NSMenuItem(title: "  Local AI: Execute (8B Tools)", action: #selector(switchToMageExecuteAction(_:)), keyEquivalent: "4")
        mageExecuteItem.tag = MenuItemTag.switchToMageExecute.rawValue
        mageExecuteItem.target = self
        mageExecuteItem.keyEquivalentModifierMask = [.command, .option]
        routerSubmenu.addItem(mageExecuteItem)

        let mageFastItem = NSMenuItem(title: "  Local AI: Fast (7B)", action: #selector(switchToMageFastAction(_:)), keyEquivalent: "5")
        mageFastItem.tag = MenuItemTag.switchToMageFast.rawValue
        mageFastItem.target = self
        mageFastItem.keyEquivalentModifierMask = [.command, .option]
        routerSubmenu.addItem(mageFastItem)

        routerItem.submenu = routerSubmenu
        menu.addItem(routerItem)

        menu.addItem(NSMenuItem.separator())

        // Chat Interface
        let chatItem = NSMenuItem(title: "💬 Chat with Local AI", action: #selector(openChatAction(_:)), keyEquivalent: "c")
        chatItem.target = self
        chatItem.tag = MenuItemTag.chatInterface.rawValue
        chatItem.keyEquivalentModifierMask = [.command, .shift]
        menu.addItem(chatItem)

        // Analytics Dashboard
        let analyticsItem = NSMenuItem(title: "📊 Usage Analytics", action: #selector(openAnalyticsAction(_:)), keyEquivalent: "a")
        analyticsItem.target = self
        analyticsItem.keyEquivalentModifierMask = [.command, .shift]
        menu.addItem(analyticsItem)

        menu.addItem(NSMenuItem.separator())

        // Quit
        let quitItem = NSMenuItem(title: "Quit Nexus Local Compute", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
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
        sendNotification(title: "Nexus Local Compute", body: "Warming up models - this may take a few minutes...")

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

    /// Toggle model load state - if loaded, unload it; if unloaded, load it
    @objc func toggleModelLoadAction(_ sender: NSMenuItem) {
        guard let modelId = sender.representedObject as? String else {
            debugLog("ERROR: toggleModelLoadAction - no modelId in representedObject")
            return
        }

        let modelName = availableModels.first { $0.modelId == modelId }?.displayName ?? modelId

        // Check if model is disabled
        if isModelDisabled(modelId) {
            sendNotification(title: "Nexus Local Compute", body: "\(modelName) is disabled. Enable it first.")
            return
        }

        if loadedModels.contains(modelId) {
            // Model is loaded - unload it
            let modelKey = modelId.replacingOccurrences(of: "mageagent:", with: "")
            debugLog("Toggle: Unloading \(modelName)")
            sendNotification(title: "Nexus Local Compute", body: "Unloading \(modelName)...")

            unloadModel(modelKey: modelKey) { [weak self] success, memoryFreed in
                guard let self = self else { return }
                if success {
                    self.loadedModels.remove(modelId)
                    let memoryStr = memoryFreed > 0 ? " (\(memoryFreed)GB freed)" : ""
                    self.sendNotification(title: "Nexus Local Compute", body: "\(modelName) unloaded\(memoryStr)")
                    self.updateModelStatusSummary()
                    self.updateModelMenuCheckmarks()
                }
                self.checkServerStatus()
            }
        } else {
            // Model is not loaded - load it
            debugLog("Toggle: Loading \(modelName)")
            sendNotification(title: "Nexus Local Compute", body: "Loading \(modelName)...")

            warmupModel(modelId: modelId) { [weak self] success in
                guard let self = self else { return }
                if success {
                    self.loadedModels.insert(modelId)
                    self.sendNotification(title: "Nexus Local Compute", body: "\(modelName) loaded!")
                    self.updateModelStatusSummary()
                    self.updateModelMenuCheckmarks()
                }
                self.checkServerStatus()
            }
        }
    }

    /// Update the model status summary in the menu
    private func updateModelStatusSummary() {
        DispatchQueue.main.async {
            guard let summaryItem = self.menu.item(withTag: 5000) else { return }

            let loadedCount = self.loadedModels.count
            let totalMemory = self.loadedModels.compactMap { modelId -> Double? in
                let key = modelId.replacingOccurrences(of: "mageagent:", with: "")
                return self.modelSizes[key] ?? self.modelSizes[modelId]
            }.reduce(0, +)

            summaryItem.title = "Loaded: \(loadedCount) model\(loadedCount == 1 ? "" : "s") (\(Int(totalMemory))GB)"
        }
    }

    @objc func loadModelAction(_ sender: NSMenuItem) {
        guard let modelId = sender.representedObject as? String else {
            debugLog("ERROR: loadModelAction - no modelId in representedObject")
            return
        }

        // Find model info
        let modelName = availableModels.first { $0.modelId == modelId }?.displayName ?? modelId
        debugLog("Load Model action triggered for: \(modelName)")

        // Check if model is disabled
        if isModelDisabled(modelId) {
            sendNotification(title: "Nexus Local Compute", body: "\(modelName) is disabled. Enable it first to load.")
            debugLog("Model \(modelName) is disabled, cannot load")
            return
        }

        // Show loading notification
        sendNotification(title: "Nexus Local Compute", body: "Loading \(modelName)...")

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
                self.sendNotification(title: "Nexus Local Compute", body: "\(modelName) loaded successfully!")
                self.debugLog("Model \(modelName) loaded into memory")

                // Update menu item to show checkmark
                DispatchQueue.main.async {
                    sender.state = .on
                }
            } else {
                self.sendNotification(title: "Nexus Local Compute", body: "Failed to load \(modelName)")
                self.debugLog("Failed to load model \(modelName)")
            }

            // Restore status display
            self.checkServerStatus()
        }
    }

    @objc func unloadModelAction(_ sender: NSMenuItem) {
        guard let modelId = sender.representedObject as? String else {
            debugLog("ERROR: unloadModelAction - no modelId in representedObject")
            return
        }

        // Find model info
        let modelName = availableModels.first { $0.modelId == modelId }?.displayName ?? modelId
        let modelKey = modelId.replacingOccurrences(of: "mageagent:", with: "")
        debugLog("Unload Model action triggered for: \(modelName)")

        // Check if model is even loaded
        if !loadedModels.contains(modelId) {
            sendNotification(title: "Nexus Local Compute", body: "\(modelName) is not currently loaded")
            return
        }

        // Show unloading notification
        sendNotification(title: "Nexus Local Compute", body: "Unloading \(modelName)...")

        // Update status to show unloading
        DispatchQueue.main.async {
            if let statusItem = self.menu.item(withTag: MenuItemTag.status.rawValue) {
                statusItem.title = "Unloading: \(modelName)..."
            }
        }

        // Unload the model via API
        unloadModel(modelKey: modelKey) { [weak self] success, memoryFreed in
            guard let self = self else { return }

            if success {
                self.loadedModels.remove(modelId)
                let memoryStr = memoryFreed > 0 ? " (\(memoryFreed)GB freed)" : ""
                self.sendNotification(title: "Nexus Local Compute", body: "\(modelName) unloaded\(memoryStr)")
                self.debugLog("Model \(modelName) unloaded from memory")

                // Update menu item to remove checkmark
                self.updateModelMenuCheckmarks()
            } else {
                self.sendNotification(title: "Nexus Local Compute", body: "Failed to unload \(modelName)")
                self.debugLog("Failed to unload model \(modelName)")
            }

            // Restore status display
            self.checkServerStatus()
        }
    }

    @objc func unloadAllModelsAction(_ sender: NSMenuItem) {
        debugLog("Unload All Models action triggered")

        if loadedModels.isEmpty {
            sendNotification(title: "Nexus Local Compute", body: "No models are currently loaded")
            return
        }

        let count = loadedModels.count
        sendNotification(title: "Nexus Local Compute", body: "Unloading \(count) models...")

        // Update status
        DispatchQueue.main.async {
            if let statusItem = self.menu.item(withTag: MenuItemTag.status.rawValue) {
                statusItem.title = "Unloading all models..."
            }
        }

        // Call unload-all endpoint
        unloadAllModels { [weak self] success, totalMemoryFreed in
            guard let self = self else { return }

            if success {
                self.loadedModels.removeAll()
                let memoryStr = totalMemoryFreed > 0 ? " (\(totalMemoryFreed)GB freed)" : ""
                self.sendNotification(title: "Nexus Local Compute", body: "All models unloaded\(memoryStr)")
                self.debugLog("All models unloaded from memory")
                self.updateModelMenuCheckmarks()
            } else {
                self.sendNotification(title: "Nexus Local Compute", body: "Failed to unload all models")
            }

            self.checkServerStatus()
        }
    }

    /// Unload a specific model via API
    private func unloadModel(modelKey: String, completion: @escaping (Bool, Int) -> Void) {
        guard let url = URL(string: "\(Config.mageagentURL)/models/unload") else {
            completion(false, 0)
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30

        let body = ["model": modelKey]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Unload model error: \(error.localizedDescription)")
                DispatchQueue.main.async { completion(false, 0) }
                return
            }

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200,
                  let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                DispatchQueue.main.async { completion(false, 0) }
                return
            }

            let memoryFreed = json["memory_freed_gb"] as? Int ?? 0
            DispatchQueue.main.async { completion(true, memoryFreed) }
        }.resume()
    }

    /// Unload all models via API
    private func unloadAllModels(completion: @escaping (Bool, Int) -> Void) {
        guard let url = URL(string: "\(Config.mageagentURL)/models/unload-all") else {
            completion(false, 0)
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 60

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Unload all models error: \(error.localizedDescription)")
                DispatchQueue.main.async { completion(false, 0) }
                return
            }

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200,
                  let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                DispatchQueue.main.async { completion(false, 0) }
                return
            }

            let totalMemory = json["total_memory_freed_gb"] as? Int ?? 0
            DispatchQueue.main.async { completion(true, totalMemory) }
        }.resume()
    }

    /// Update model menu items to show visual status indicators
    private func updateModelMenuCheckmarks() {
        DispatchQueue.main.async {
            // Update individual model menu items with visual indicators
            for (index, model) in self.availableModels.enumerated() {
                let tag = 1000 + index
                if let menuItem = self.menu.item(withTag: tag) {
                    let isLoaded = self.loadedModels.contains(model.modelId)
                    let isDisabled = self.disabledModels.contains(model.modelId)

                    // Update title with visual indicator
                    var title: String
                    if isDisabled {
                        title = "⊘ \(model.displayName) [DISABLED]"
                        menuItem.attributedTitle = NSAttributedString(
                            string: title,
                            attributes: [.foregroundColor: NSColor.disabledControlTextColor]
                        )
                        menuItem.isEnabled = false
                    } else if isLoaded {
                        title = "✓ \(model.displayName)"
                        menuItem.attributedTitle = nil
                        menuItem.title = title
                        menuItem.isEnabled = true
                    } else {
                        title = "○ \(model.displayName)"
                        menuItem.attributedTitle = nil
                        menuItem.title = title
                        menuItem.isEnabled = true
                    }

                    menuItem.state = isLoaded ? .on : .off
                }
            }

            // Also update the status summary
            self.updateModelStatusSummary()
        }
    }

    // MARK: - Model Disable Feature (Persisted)

    /// Load disabled models from UserDefaults
    private func loadDisabledModels() {
        if let savedDisabled = UserDefaults.standard.array(forKey: disabledModelsKey) as? [String] {
            disabledModels = Set(savedDisabled)
            debugLog("Loaded \(disabledModels.count) disabled models: \(disabledModels)")
        }
    }

    /// Save disabled models to UserDefaults
    private func saveDisabledModels() {
        UserDefaults.standard.set(Array(disabledModels), forKey: disabledModelsKey)
        debugLog("Saved \(disabledModels.count) disabled models")
    }

    /// Toggle model disabled state (action for menu item)
    @objc func toggleModelDisabledAction(_ sender: NSMenuItem) {
        guard let modelId = sender.representedObject as? String else {
            debugLog("ERROR: toggleModelDisabledAction - no modelId in representedObject")
            return
        }

        let modelName = availableModels.first { $0.modelId == modelId }?.displayName ?? modelId

        if disabledModels.contains(modelId) {
            // Re-enable the model
            disabledModels.remove(modelId)
            sendNotification(title: "Nexus Local Compute", body: "\(modelName) enabled")
            debugLog("Model \(modelName) enabled")
        } else {
            // Disable the model
            disabledModels.insert(modelId)

            // Also unload if currently loaded
            if loadedModels.contains(modelId) {
                let modelKey = modelId.replacingOccurrences(of: "mageagent:", with: "")
                unloadModel(modelKey: modelKey) { success, _ in
                    if success {
                        self.loadedModels.remove(modelId)
                    }
                }
            }

            sendNotification(title: "Nexus Local Compute", body: "\(modelName) disabled")
            debugLog("Model \(modelName) disabled")
        }

        // Persist the change
        saveDisabledModels()

        // Update menu items
        updateModelMenuAppearance()
    }

    /// Update model menu items to show disabled state
    private func updateModelMenuAppearance() {
        DispatchQueue.main.async {
            for (index, model) in self.availableModels.enumerated() {
                // Update load model item
                let loadTag = 1000 + index
                if let loadItem = self.menu.item(withTag: loadTag) {
                    let isDisabled = self.disabledModels.contains(model.modelId)
                    let isLoaded = self.loadedModels.contains(model.modelId)

                    if isDisabled {
                        // Show strikethrough or grayed out for disabled
                        let attributedTitle = NSAttributedString(
                            string: "⊘ \(model.displayName) [DISABLED]",
                            attributes: [
                                .foregroundColor: NSColor.disabledControlTextColor,
                                .strikethroughStyle: NSUnderlineStyle.single.rawValue
                            ]
                        )
                        loadItem.attributedTitle = attributedTitle
                        loadItem.isEnabled = false
                        loadItem.state = .off
                    } else {
                        // Normal state
                        loadItem.title = model.displayName
                        loadItem.attributedTitle = nil
                        loadItem.isEnabled = true
                        loadItem.state = isLoaded ? .on : .off
                    }
                }

                // Update unload model item
                let unloadTag = 3000 + index
                if let unloadItem = self.menu.item(withTag: unloadTag) {
                    let isDisabled = self.disabledModels.contains(model.modelId)
                    unloadItem.isEnabled = !isDisabled
                }

                // Update disable toggle item
                let disableTag = 4000 + index
                if let disableItem = self.menu.item(withTag: disableTag) {
                    let isDisabled = self.disabledModels.contains(model.modelId)
                    disableItem.title = isDisabled ? "Enable \(model.displayName)" : "Disable \(model.displayName)"
                    disableItem.state = isDisabled ? .on : .off
                }
            }
        }
    }

    /// Check if a model is disabled before loading
    private func isModelDisabled(_ modelId: String) -> Bool {
        return disabledModels.contains(modelId)
    }

    // MARK: - Model Presets Feature

    /// Load custom presets from UserDefaults
    private func loadCustomPresets() {
        if let data = UserDefaults.standard.data(forKey: customPresetsKey),
           let presets = try? JSONDecoder().decode([ModelPreset].self, from: data) {
            customPresets = presets
            debugLog("Loaded \(customPresets.count) custom presets")
        }
    }

    /// Save custom presets to UserDefaults
    private func saveCustomPresets() {
        if let data = try? JSONEncoder().encode(customPresets) {
            UserDefaults.standard.set(data, forKey: customPresetsKey)
            debugLog("Saved \(customPresets.count) custom presets")
        }
    }

    /// Apply a preset (unload all, then load preset models)
    @objc func applyPresetAction(_ sender: NSMenuItem) {
        guard let preset = sender.representedObject as? ModelPreset else {
            debugLog("ERROR: applyPresetAction - no preset in representedObject")
            return
        }

        debugLog("Applying preset: \(preset.name)")
        currentPresetName = preset.name

        // First unload all current models
        let modelsToUnload = loadedModels.filter { !preset.models.contains($0) }
        let modelsToLoad = preset.models.filter { !loadedModels.contains($0) && !disabledModels.contains($0) }

        sendNotification(title: "Nexus Local Compute", body: "Applying preset: \(preset.name)")

        // Unload models not in preset
        for modelId in modelsToUnload {
            let modelKey = modelId.replacingOccurrences(of: "mageagent:", with: "")
            unloadModel(modelKey: modelKey) { success, _ in
                if success {
                    self.loadedModels.remove(modelId)
                }
            }
        }

        // Load preset models (with slight delay to allow unloads to complete)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.loadPresetModels(preset: preset, modelIds: Array(modelsToLoad), index: 0)
        }
    }

    /// Load models for a preset sequentially
    private func loadPresetModels(preset: ModelPreset, modelIds: [String], index: Int) {
        guard index < modelIds.count else {
            // All models loaded
            sendNotification(title: "Nexus Local Compute", body: "Preset '\(preset.name)' active!")
            updateModelMenuCheckmarks()
            return
        }

        let modelId = modelIds[index]
        let modelName = availableModels.first { $0.modelId == modelId }?.displayName ?? modelId

        warmupModel(modelId: modelId) { [weak self] success in
            guard let self = self else { return }

            if success {
                self.loadedModels.insert(modelId)
            } else {
                self.debugLog("Failed to load \(modelName) for preset")
            }

            // Continue to next model
            self.loadPresetModels(preset: preset, modelIds: modelIds, index: index + 1)
        }
    }

    /// Save current loaded models as a new preset
    @objc func saveCurrentAsPresetAction(_ sender: NSMenuItem) {
        // Create alert with text field for preset name
        let alert = NSAlert()
        alert.messageText = "Save Current Configuration as Preset"
        alert.informativeText = "Enter a name for this preset.\n\nCurrently loaded: \(loadedModels.count) models"
        alert.addButton(withTitle: "Save")
        alert.addButton(withTitle: "Cancel")

        let textField = NSTextField(frame: NSRect(x: 0, y: 0, width: 250, height: 24))
        textField.stringValue = "My Preset"
        textField.placeholderString = "Preset name"
        alert.accessoryView = textField

        alert.window.initialFirstResponder = textField

        let response = alert.runModal()
        if response == .alertFirstButtonReturn {
            let presetName = textField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !presetName.isEmpty else { return }

            let newPreset = ModelPreset(
                name: presetName,
                models: Array(loadedModels),
                isBuiltIn: false
            )

            customPresets.append(newPreset)
            saveCustomPresets()
            sendNotification(title: "Nexus Local Compute", body: "Preset '\(presetName)' saved!")
        }
    }

    /// Delete a custom preset
    @objc func deletePresetAction(_ sender: NSMenuItem) {
        guard let presetName = sender.representedObject as? String else { return }

        customPresets.removeAll { $0.name == presetName }
        saveCustomPresets()
        sendNotification(title: "Nexus Local Compute", body: "Preset '\(presetName)' deleted")
    }

    // MARK: - Pattern Selection Actions

    @objc func selectPatternAction(_ sender: NSMenuItem) {
        guard let pattern = sender.representedObject as? PatternInfo else {
            debugLog("ERROR: selectPatternAction - no PatternInfo in representedObject")
            return
        }

        debugLog("Pattern selection triggered: \(pattern.displayName)")

        // Check if any required models are disabled
        let disabledRequired = pattern.requiredModels.filter { disabledModels.contains($0) }
        if !disabledRequired.isEmpty {
            let disabledNames = disabledRequired.compactMap { modelId in
                availableModels.first { $0.modelId == modelId }?.displayName
            }.joined(separator: ", ")

            sendNotification(
                title: "Nexus Local Compute",
                body: "Pattern '\(pattern.displayName)' requires disabled model(s):\n\(disabledNames)\n\nEnable them first to use this pattern."
            )
            debugLog("Pattern requires disabled models: \(disabledRequired)")
            return
        }

        // Update selected pattern
        selectedPattern = pattern.patternId

        // Check which required models are not yet loaded (excluding disabled)
        let missingModels = pattern.requiredModels.filter { !loadedModels.contains($0) && !disabledModels.contains($0) }

        if missingModels.isEmpty {
            // All models already loaded
            sendNotification(title: "Nexus Local Compute", body: "Pattern '\(pattern.displayName)' active - all required models already loaded!")
            updatePatternMenuCheckmarks(selectedPatternId: pattern.patternId)
        } else {
            // Need to load missing models
            let modelNames = missingModels.compactMap { modelId in
                availableModels.first { $0.modelId == modelId }?.displayName
            }.joined(separator: ", ")

            sendNotification(title: "Nexus Local Compute", body: "Loading models for '\(pattern.displayName)':\n\(modelNames)")

            // Load missing models sequentially
            loadModelsForPattern(pattern: pattern, modelIds: missingModels, index: 0)
        }
    }

    /// Load required models for a pattern sequentially
    private func loadModelsForPattern(pattern: PatternInfo, modelIds: [String], index: Int) {
        guard index < modelIds.count else {
            // All models loaded
            debugLog("All models for pattern '\(pattern.displayName)' loaded successfully")
            sendNotification(title: "Nexus Local Compute", body: "Pattern '\(pattern.displayName)' ready!")
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
            sendNotification(title: "Nexus Local Compute", body: "All models loaded into memory!")
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
                self.sendNotification(title: "Nexus Local Compute", body: successMessage)
                // Delay status check to allow server to start/stop
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    self.checkServerStatus()
                }
            } else {
                let errorDetail = output.isEmpty ? "" : "\n\(output.prefix(200))"
                self.sendNotification(title: "Nexus Local Compute", body: "\(failureMessage)\(errorDetail)")
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

    /// Settings floating panel
    private var settingsPanel: NSPanel?

    /// Chat window and components
    private var chatWindow: NSWindow?
    private var chatInputField: NSTextField?
    private var chatOutputTextView: NSTextView?
    private var chatMessages: [(role: String, content: String)] = []
    private var isChatLoading: Bool = false
    private var pendingToolCalls: [[String: Any]] = []
    private var chatToolExecutionEnabled: Bool = true

    /// File attachment support
    private var attachedFiles: [(url: URL, name: String)] = []
    private var attachmentContainerView: NSView?
    private var attachmentStackView: NSStackView?
    private var memoryProgressBar: NSProgressIndicator?
    private var cpuProgressBar: NSProgressIndicator?
    private var gpuProgressBar: NSProgressIndicator?

    /// Analytics window
    private var analyticsWindow: NSWindow?
    private var analyticsTextView: NSTextView?

    /// Show streaming test results in a window
    private func showTestResultsWindow() {
        // Create window
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 600, height: 450),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Nexus Local Compute - Test Results"
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
        appendTestResult("Nexus Local Compute - Test Suite\n", color: .systemBlue)
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
                sendNotification(title: "Nexus Tests", body: "All \(total) tests passed!")
            } else {
                appendTestResult("\(failed) test(s) failed\n", color: .systemRed)
                updateTestStatus(summary, color: .systemRed)
                sendNotification(title: "Nexus Tests", body: "\(failed) of \(total) tests failed")
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

        // If panel already exists, toggle visibility
        if let existingPanel = settingsPanel {
            if existingPanel.isVisible {
                closeSettingsPanel()
            } else {
                showSettingsPanelWithAnimation()
            }
            return
        }

        createSettingsPanel()
        showSettingsPanelWithAnimation()
    }

    private func createSettingsPanel() {
        let panelWidth: CGFloat = 320
        let panelHeight: CGFloat = 680

        // Create floating panel that looks like the menu (now resizable)
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: panelWidth, height: panelHeight),
            styleMask: [.titled, .closable, .resizable, .nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        panel.minSize = NSSize(width: 280, height: 400)
        panel.maxSize = NSSize(width: 600, height: 1200)

        panel.title = "Nexus Local Compute"
        panel.titleVisibility = .hidden
        panel.titlebarAppearsTransparent = true
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.isMovableByWindowBackground = true
        panel.isReleasedWhenClosed = false
        panel.hidesOnDeactivate = false
        panel.becomesKeyOnlyIfNeeded = true

        // Match menu dark background
        panel.backgroundColor = NSColor(calibratedWhite: 0.18, alpha: 0.98)

        // Create content view
        let contentView = NSView(frame: NSRect(x: 0, y: 0, width: panelWidth, height: panelHeight))

        var yOffset: CGFloat = panelHeight - 15

        // Get current system data
        let memInfo = getMemoryInfo()
        let cpuInfo = getCPUUsage()
        let gpuInfo = getGPUInfo()

        // Determine overall pressure for header indicator
        var overallPressure: SystemPressure = .nominal
        if memInfo.pressure == .critical || cpuInfo.pressure == .critical || gpuInfo.pressure == .critical {
            overallPressure = .critical
        } else if memInfo.pressure == .warning || cpuInfo.pressure == .warning || gpuInfo.pressure == .warning {
            overallPressure = .warning
        }

        // Status line - matches menu
        let statusText = isServerRunning ? "Status: Running (v2.3.0)" : "Status: Stopped"
        let statusLabel = NSTextField(labelWithString: statusText)
        statusLabel.frame = NSRect(x: 15, y: yOffset - 18, width: panelWidth - 30, height: 18)
        statusLabel.font = NSFont.systemFont(ofSize: 13)
        statusLabel.textColor = .lightGray
        statusLabel.tag = 2001
        contentView.addSubview(statusLabel)
        yOffset -= 28

        // Separator
        let sep1 = createSeparator(y: yOffset, width: panelWidth)
        contentView.addSubview(sep1)
        yOffset -= 12

        // System Resources Header with indicator
        let resourcesHeader = NSTextField(labelWithString: "")
        resourcesHeader.frame = NSRect(x: 15, y: yOffset - 20, width: panelWidth - 30, height: 20)
        let headerAttr = NSMutableAttributedString(string: "● ", attributes: [.foregroundColor: overallPressure.color])
        headerAttr.append(NSAttributedString(string: "System Resources", attributes: [
            .foregroundColor: NSColor.white,
            .font: NSFont.systemFont(ofSize: 13, weight: .semibold)
        ]))
        resourcesHeader.attributedStringValue = headerAttr
        resourcesHeader.tag = 2002
        contentView.addSubview(resourcesHeader)
        yOffset -= 28

        // Memory line with indicator
        let memoryLine = createResourceLine(
            indicator: memInfo.pressure.color,
            text: String(format: "Memory: %.1f / %.1f GB (%.0f%%)", memInfo.usedGB, memInfo.totalGB, memInfo.percentUsed),
            y: yOffset,
            width: panelWidth,
            tag: 2003
        )
        contentView.addSubview(memoryLine)
        yOffset -= 24

        // ===== ACTIVITY MONITOR STYLE MEMORY PRESSURE GRAPH =====
        let graphWidth: CGFloat = panelWidth - 60
        let graphHeight: CGFloat = 50
        let graphX: CGFloat = 30

        // Container for the graph section
        let graphContainer = NSView(frame: NSRect(x: graphX, y: yOffset - graphHeight - 4, width: graphWidth, height: graphHeight + 16))
        graphContainer.identifier = NSUserInterfaceItemIdentifier("panelMemPressureContainer")
        contentView.addSubview(graphContainer)

        // Title label "MEMORY PRESSURE" like Activity Monitor
        let titleLabel = NSTextField(labelWithString: "MEMORY PRESSURE")
        titleLabel.frame = NSRect(x: 0, y: graphHeight + 2, width: graphWidth, height: 12)
        titleLabel.font = NSFont.systemFont(ofSize: 9, weight: .medium)
        titleLabel.textColor = NSColor.lightGray
        titleLabel.alignment = .center
        graphContainer.addSubview(titleLabel)

        // Graph background (dark rectangle)
        let graphBg = NSView(frame: NSRect(x: 0, y: 0, width: graphWidth, height: graphHeight))
        graphBg.wantsLayer = true
        graphBg.layer?.backgroundColor = NSColor(white: 0.08, alpha: 1.0).cgColor
        graphBg.layer?.cornerRadius = 3
        graphBg.layer?.borderWidth = 1
        graphBg.layer?.borderColor = NSColor(white: 0.25, alpha: 1.0).cgColor
        graphBg.identifier = NSUserInterfaceItemIdentifier("panelMemPressureGraphBg")
        graphContainer.addSubview(graphBg)

        // Graph line view
        let graphView = NSView(frame: NSRect(x: 0, y: 0, width: graphWidth, height: graphHeight))
        graphView.wantsLayer = true
        graphView.identifier = NSUserInterfaceItemIdentifier("panelMemPressureGraph")
        graphBg.addSubview(graphView)

        // Initialize history if empty
        if memoryPressureHistory.isEmpty {
            memoryPressureHistory = Array(repeating: memInfo.percentUsed, count: maxHistorySamples)
        }

        // Draw initial graph
        drawMemoryPressureGraph(in: graphView)

        yOffset -= (graphHeight + 24)

        // CPU line with indicator
        let cpuLine = createResourceLine(
            indicator: cpuInfo.pressure.color,
            text: String(format: "CPU: %.1f%%", cpuInfo.usage),
            y: yOffset,
            width: panelWidth,
            tag: 2004
        )
        contentView.addSubview(cpuLine)
        yOffset -= 24

        // GPU line with indicator
        let gpuLine = createResourceLine(
            indicator: gpuInfo.pressure.color,
            text: String(format: "GPU: %.0f%%", gpuInfo.usagePercent),
            y: yOffset,
            width: panelWidth,
            tag: 2005
        )
        contentView.addSubview(gpuLine)
        yOffset -= 24

        // Neural Engine line
        let neuralLine = createResourceLine(
            indicator: NSColor.systemBlue,
            text: "Neural Engine: \(neuralEngineTOPS) TOPS",
            y: yOffset,
            width: panelWidth,
            tag: 2006
        )
        contentView.addSubview(neuralLine)
        yOffset -= 24

        // Tokens line
        let tokensText = lastThroughput.tokensPerSec > 0
            ? String(format: "Tokens: %.1f tok/s (%d total)", lastThroughput.tokensPerSec, lastThroughput.totalTokens)
            : "Tokens: Idle (\(lastThroughput.totalTokens) total)"
        let tokensLine = createResourceLine(
            indicator: NSColor.gray,
            text: tokensText,
            y: yOffset,
            width: panelWidth,
            tag: 2007
        )
        contentView.addSubview(tokensLine)
        yOffset -= 20

        // Separator
        let sep2 = createSeparator(y: yOffset, width: panelWidth)
        contentView.addSubview(sep2)
        yOffset -= 12

        // Server Controls Section
        let controlsTitle = NSTextField(labelWithString: "Server Controls")
        controlsTitle.frame = NSRect(x: 15, y: yOffset - 18, width: panelWidth - 30, height: 18)
        controlsTitle.font = NSFont.systemFont(ofSize: 13, weight: .semibold)
        controlsTitle.textColor = .white
        contentView.addSubview(controlsTitle)
        yOffset -= 28

        // Control buttons
        let buttonWidth: CGFloat = (panelWidth - 50) / 2
        let buttonHeight: CGFloat = 28

        let startBtn = NSButton(title: "Start Server", target: self, action: #selector(startServerAction(_:)))
        startBtn.frame = NSRect(x: 15, y: yOffset - buttonHeight, width: buttonWidth, height: buttonHeight)
        startBtn.bezelStyle = .rounded
        contentView.addSubview(startBtn)

        let stopBtn = NSButton(title: "Stop Server", target: self, action: #selector(stopServerAction(_:)))
        stopBtn.frame = NSRect(x: 25 + buttonWidth, y: yOffset - buttonHeight, width: buttonWidth, height: buttonHeight)
        stopBtn.bezelStyle = .rounded
        contentView.addSubview(stopBtn)
        yOffset -= (buttonHeight + 8)

        let restartBtn = NSButton(title: "Restart Server", target: self, action: #selector(restartServerAction(_:)))
        restartBtn.frame = NSRect(x: 15, y: yOffset - buttonHeight, width: buttonWidth, height: buttonHeight)
        restartBtn.bezelStyle = .rounded
        contentView.addSubview(restartBtn)

        let warmupBtn = NSButton(title: "Warmup Models", target: self, action: #selector(warmupModelsAction(_:)))
        warmupBtn.frame = NSRect(x: 25 + buttonWidth, y: yOffset - buttonHeight, width: buttonWidth, height: buttonHeight)
        warmupBtn.bezelStyle = .rounded
        contentView.addSubview(warmupBtn)
        yOffset -= (buttonHeight + 15)

        // Separator
        let sep3 = createSeparator(y: yOffset, width: panelWidth)
        contentView.addSubview(sep3)
        yOffset -= 12

        // Quick Actions Section
        let actionsTitle = NSTextField(labelWithString: "Quick Actions")
        actionsTitle.frame = NSRect(x: 15, y: yOffset - 18, width: panelWidth - 30, height: 18)
        actionsTitle.font = NSFont.systemFont(ofSize: 13, weight: .semibold)
        actionsTitle.textColor = .white
        contentView.addSubview(actionsTitle)
        yOffset -= 28

        let docsBtn = NSButton(title: "Open API Docs", target: self, action: #selector(openDocsAction(_:)))
        docsBtn.frame = NSRect(x: 15, y: yOffset - buttonHeight, width: buttonWidth, height: buttonHeight)
        docsBtn.bezelStyle = .rounded
        contentView.addSubview(docsBtn)

        let logsBtn = NSButton(title: "View Logs", target: self, action: #selector(viewLogsAction(_:)))
        logsBtn.frame = NSRect(x: 25 + buttonWidth, y: yOffset - buttonHeight, width: buttonWidth, height: buttonHeight)
        logsBtn.bezelStyle = .rounded
        contentView.addSubview(logsBtn)
        yOffset -= (buttonHeight + 8)

        let testBtn = NSButton(title: "Run Test", target: self, action: #selector(runTestAction(_:)))
        testBtn.frame = NSRect(x: 15, y: yOffset - buttonHeight, width: buttonWidth, height: buttonHeight)
        testBtn.bezelStyle = .rounded
        contentView.addSubview(testBtn)

        // Offline Mode Toggle
        let offlineBtn = NSButton(title: isOfflineMode ? "Go Online" : "Go Offline", target: self, action: #selector(toggleOfflineModeFromPanel(_:)))
        offlineBtn.frame = NSRect(x: 25 + buttonWidth, y: yOffset - buttonHeight, width: buttonWidth, height: buttonHeight)
        offlineBtn.bezelStyle = .rounded
        offlineBtn.tag = 2020
        contentView.addSubview(offlineBtn)
        yOffset -= (buttonHeight + 15)

        // Separator
        let sep4 = createSeparator(y: yOffset, width: panelWidth)
        contentView.addSubview(sep4)
        yOffset -= 12

        // Loaded Models Section
        let modelsTitle = NSTextField(labelWithString: "Loaded Models")
        modelsTitle.frame = NSRect(x: 15, y: yOffset - 18, width: panelWidth - 30, height: 18)
        modelsTitle.font = NSFont.systemFont(ofSize: 13, weight: .semibold)
        modelsTitle.textColor = .white
        contentView.addSubview(modelsTitle)
        yOffset -= 26

        let models = [
            ("Primary", "Qwen-72B Q8", loadedModels.contains("primary") || loadedModels.contains("mageagent:primary")),
            ("Tools", "Hermes-3 8B Q8", loadedModels.contains("tools") || loadedModels.contains("mageagent:tools")),
            ("Validator", "Qwen-Coder 7B", loadedModels.contains("validator") || loadedModels.contains("mageagent:validator")),
            ("Competitor", "Qwen-Coder 32B", loadedModels.contains("competitor") || loadedModels.contains("mageagent:competitor"))
        ]

        for (name, desc, loaded) in models {
            let line = createResourceLine(
                indicator: loaded ? NSColor.systemGreen : NSColor.systemGray,
                text: "\(name): \(desc)",
                y: yOffset,
                width: panelWidth,
                tag: 0
            )
            line.alphaValue = loaded ? 1.0 : 0.5
            contentView.addSubview(line)
            yOffset -= 22
        }

        // Close button at bottom
        let closeButton = NSButton(title: "Close Panel", target: self, action: #selector(closeSettingsFromButton))
        closeButton.frame = NSRect(x: (panelWidth - 120) / 2, y: 15, width: 120, height: 32)
        closeButton.bezelStyle = .rounded
        closeButton.keyEquivalent = "\u{1b}"
        contentView.addSubview(closeButton)

        panel.contentView = contentView
        settingsPanel = panel

        // Start the update loop
        startSettingsPanelUpdateLoop()
    }

    private func createSeparator(y: CGFloat, width: CGFloat) -> NSBox {
        let sep = NSBox(frame: NSRect(x: 10, y: y, width: width - 20, height: 1))
        sep.boxType = .separator
        return sep
    }

    /// Creates Activity Monitor style memory pressure GRAPH view for the dropdown menu
    /// Shows a real-time scrolling line graph like Activity Monitor's Memory Pressure display
    private func createMemoryPressureBarView() -> NSView {
        let viewWidth: CGFloat = 260
        let viewHeight: CGFloat = 70  // Increased height for title + graph
        let graphWidth: CGFloat = viewWidth - 50
        let graphHeight: CGFloat = 40
        let graphX: CGFloat = 25
        let graphY: CGFloat = 6  // Graph at bottom

        let containerView = NSView(frame: NSRect(x: 0, y: 0, width: viewWidth, height: viewHeight))
        containerView.identifier = NSUserInterfaceItemIdentifier("memPressureContainer")

        // Title label "MEMORY PRESSURE" like Activity Monitor - positioned above graph
        let titleLabel = NSTextField(labelWithString: "MEMORY PRESSURE")
        titleLabel.frame = NSRect(x: graphX, y: graphY + graphHeight + 4, width: graphWidth, height: 14)
        titleLabel.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        titleLabel.textColor = NSColor.lightGray
        titleLabel.alignment = .center
        containerView.addSubview(titleLabel)

        // Graph background (dark rectangle)
        let graphBg = NSView(frame: NSRect(x: graphX, y: graphY, width: graphWidth, height: graphHeight))
        graphBg.wantsLayer = true
        graphBg.layer?.backgroundColor = NSColor(white: 0.08, alpha: 1.0).cgColor
        graphBg.layer?.cornerRadius = 3
        graphBg.layer?.borderWidth = 1
        graphBg.layer?.borderColor = NSColor(white: 0.3, alpha: 1.0).cgColor
        graphBg.identifier = NSUserInterfaceItemIdentifier("memPressureGraphBg")
        containerView.addSubview(graphBg)

        // Graph line view (will be drawn with CAShapeLayer)
        let graphView = NSView(frame: NSRect(x: 0, y: 0, width: graphWidth, height: graphHeight))
        graphView.wantsLayer = true
        graphView.identifier = NSUserInterfaceItemIdentifier("memPressureGraph")
        graphBg.addSubview(graphView)

        // Initialize history if empty
        if memoryPressureHistory.isEmpty {
            let memInfo = getMemoryInfo()
            memoryPressureHistory = Array(repeating: memInfo.percentUsed, count: maxHistorySamples)
        }

        // Draw initial graph
        drawMemoryPressureGraph(in: graphView)

        return containerView
    }

    /// Draws the memory pressure line graph
    private func drawMemoryPressureGraph(in graphView: NSView) {
        guard graphView.layer != nil else { return }

        // Remove old shape layer
        graphView.layer?.sublayers?.removeAll(where: { $0 is CAShapeLayer })

        let width = graphView.frame.width
        let height = graphView.frame.height
        let padding: CGFloat = 2

        // Create path for the line graph
        let path = NSBezierPath()
        let fillPath = NSBezierPath()

        let dataPoints = memoryPressureHistory
        guard dataPoints.count > 1 else { return }

        let xStep = (width - padding * 2) / CGFloat(maxHistorySamples - 1)

        // Start fill path at bottom left
        fillPath.move(to: NSPoint(x: padding, y: padding))

        for (index, value) in dataPoints.enumerated() {
            let x = padding + CGFloat(index) * xStep
            let y = padding + (height - padding * 2) * CGFloat(value / 100.0)

            if index == 0 {
                path.move(to: NSPoint(x: x, y: y))
                fillPath.line(to: NSPoint(x: x, y: y))
            } else {
                path.line(to: NSPoint(x: x, y: y))
                fillPath.line(to: NSPoint(x: x, y: y))
            }
        }

        // Close fill path
        let lastX = padding + CGFloat(dataPoints.count - 1) * xStep
        fillPath.line(to: NSPoint(x: lastX, y: padding))
        fillPath.close()

        // Determine color based on current pressure
        let currentPressure = dataPoints.last ?? 0
        let lineColor: NSColor
        let fillColor: NSColor
        if currentPressure >= 90 {
            lineColor = .systemRed
            fillColor = NSColor.systemRed.withAlphaComponent(0.3)
        } else if currentPressure >= 75 {
            lineColor = .systemYellow
            fillColor = NSColor.systemYellow.withAlphaComponent(0.3)
        } else {
            lineColor = .systemGreen
            fillColor = NSColor.systemGreen.withAlphaComponent(0.3)
        }

        // Create fill layer
        let fillLayer = CAShapeLayer()
        fillLayer.path = fillPath.cgPath
        fillLayer.fillColor = fillColor.cgColor
        fillLayer.strokeColor = nil
        graphView.layer?.addSublayer(fillLayer)

        // Create line layer
        let lineLayer = CAShapeLayer()
        lineLayer.path = path.cgPath
        lineLayer.strokeColor = lineColor.cgColor
        lineLayer.fillColor = nil
        lineLayer.lineWidth = 1.5
        lineLayer.lineCap = .round
        lineLayer.lineJoin = .round
        graphView.layer?.addSublayer(lineLayer)
    }

    /// Updates the Activity Monitor style memory pressure GRAPH in the dropdown menu
    private func updateMenuMemoryPressureBar() {
        let memInfo = getMemoryInfo()

        // Add new data point to history
        memoryPressureHistory.append(memInfo.percentUsed)
        if memoryPressureHistory.count > maxHistorySamples {
            memoryPressureHistory.removeFirst()
        }

        // Find and update the graph
        guard let barItem = menu.item(withTag: 610),
              let containerView = barItem.view else { return }

        // Find graph view and redraw
        func findGraphView(in view: NSView) -> NSView? {
            if view.identifier == NSUserInterfaceItemIdentifier("memPressureGraph") {
                return view
            }
            for subview in view.subviews {
                if let found = findGraphView(in: subview) {
                    return found
                }
            }
            return nil
        }

        if let graphView = findGraphView(in: containerView) {
            drawMemoryPressureGraph(in: graphView)
        }
    }

    private func createResourceLine(indicator: NSColor, text: String, y: CGFloat, width: CGFloat, tag: Int) -> NSTextField {
        let field = NSTextField(labelWithString: "")
        field.frame = NSRect(x: 20, y: y - 18, width: width - 40, height: 18)

        let attr = NSMutableAttributedString(string: "● ", attributes: [.foregroundColor: indicator])
        attr.append(NSAttributedString(string: text, attributes: [
            .foregroundColor: NSColor.lightGray,
            .font: NSFont.systemFont(ofSize: 12)
        ]))
        field.attributedStringValue = attr
        field.tag = tag
        return field
    }

    @objc private func toggleOfflineModeFromPanel(_ sender: NSButton) {
        let command = isOfflineMode ? "disable" : "enable"
        runOfflineModeScript(command) { [weak self] success, _ in
            guard let self = self else { return }
            if success {
                self.isOfflineMode = !self.isOfflineMode
                DispatchQueue.main.async {
                    sender.title = self.isOfflineMode ? "Go Online" : "Go Offline"
                    // Update menu item too
                    if let menuItem = self.menu.item(withTag: MenuItemTag.offlineMode.rawValue) {
                        menuItem.state = self.isOfflineMode ? .on : .off
                        menuItem.title = self.isOfflineMode ? "Offline Mode (Active)" : "Offline Mode"
                    }
                }
                let message = self.isOfflineMode
                    ? "Offline mode enabled"
                    : "Online mode restored"
                self.sendNotification(title: "Nexus Local Compute", body: message)
            }
        }
    }

    private func startSettingsPanelUpdateLoop() {
        // Update immediately and then every second
        updateSettingsPanelBars()
    }

    @objc func popOutDashboardAction(_ sender: NSMenuItem) {
        debugLog("Pop Out Dashboard action triggered")

        // Close the menu first
        menu.cancelTracking()

        // Small delay to let menu close, then show the panel
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            guard let self = self else { return }

            // If panel doesn't exist, create it
            if self.settingsPanel == nil {
                self.createSettingsPanel()
            }

            self.showSettingsPanelWithAnimation()
        }
    }

    private func showSettingsPanelWithAnimation() {
        guard let panel = settingsPanel else { return }

        // Position near menu bar (top right of screen)
        if let screen = NSScreen.main {
            let screenFrame = screen.visibleFrame
            let panelFrame = panel.frame

            // Start position: just above the visible area (off screen)
            let startX = screenFrame.maxX - panelFrame.width - 20
            let startY = screenFrame.maxY + panelFrame.height

            // End position: visible below menu bar
            let endY = screenFrame.maxY - panelFrame.height - 10

            panel.setFrameOrigin(NSPoint(x: startX, y: startY))
            panel.alphaValue = 0
            panel.makeKeyAndOrderFront(nil)

            // Animate sliding down
            NSAnimationContext.runAnimationGroup { context in
                context.duration = 0.3
                context.timingFunction = CAMediaTimingFunction(name: .easeOut)
                panel.animator().setFrameOrigin(NSPoint(x: startX, y: endY))
                panel.animator().alphaValue = 1.0
            }
        }
    }

    private func closeSettingsPanel() {
        guard let panel = settingsPanel else { return }

        if let screen = NSScreen.main {
            let screenFrame = screen.visibleFrame
            let panelFrame = panel.frame

            // Animate sliding up and fading out
            let endY = screenFrame.maxY + panelFrame.height

            NSAnimationContext.runAnimationGroup({ context in
                context.duration = 0.25
                context.timingFunction = CAMediaTimingFunction(name: .easeIn)
                panel.animator().setFrameOrigin(NSPoint(x: panelFrame.origin.x, y: endY))
                panel.animator().alphaValue = 0
            }) {
                panel.orderOut(nil)
            }
        }
    }

    @objc private func closeSettingsFromButton() {
        closeSettingsPanel()
    }

    @objc private func openLogsFromSettings() {
        let logsFolder = (Config.logFile as NSString).deletingLastPathComponent
        NSWorkspace.shared.open(URL(fileURLWithPath: logsFolder))
    }

    private func updateSettingsPanelBars() {
        guard settingsPanel?.isVisible == true else { return }

        // Get current system stats
        let memInfo = getMemoryInfo()
        let cpuInfo = getCPUUsage()
        let gpuInfo = getGPUInfo()

        DispatchQueue.main.async { [weak self] in
            guard let self = self, let panel = self.settingsPanel, panel.isVisible,
                  let contentView = panel.contentView else { return }

            // Find view helper
            func findView(identifier: String, in view: NSView) -> NSView? {
                if view.identifier == NSUserInterfaceItemIdentifier(identifier) {
                    return view
                }
                for subview in view.subviews {
                    if let found = findView(identifier: identifier, in: subview) {
                        return found
                    }
                }
                return nil
            }

            // Update the memory pressure GRAPH (not bar)
            if let graphView = findView(identifier: "panelMemPressureGraph", in: contentView) {
                self.drawMemoryPressureGraph(in: graphView)
            }

            // Update text labels
            if let memLabel = contentView.viewWithTag(2003) as? NSTextField {
                let attr = NSMutableAttributedString(string: "● ", attributes: [.foregroundColor: memInfo.pressure.color])
                attr.append(NSAttributedString(string: String(format: "Memory: %.1f / %.1f GB (%.0f%%)", memInfo.usedGB, memInfo.totalGB, memInfo.percentUsed), attributes: [
                    .foregroundColor: NSColor.lightGray,
                    .font: NSFont.systemFont(ofSize: 12)
                ]))
                memLabel.attributedStringValue = attr
            }

            if let cpuLabel = contentView.viewWithTag(2004) as? NSTextField {
                let attr = NSMutableAttributedString(string: "● ", attributes: [.foregroundColor: cpuInfo.pressure.color])
                attr.append(NSAttributedString(string: String(format: "CPU: %.1f%%", cpuInfo.usage), attributes: [
                    .foregroundColor: NSColor.lightGray,
                    .font: NSFont.systemFont(ofSize: 12)
                ]))
                cpuLabel.attributedStringValue = attr
            }

            if let gpuLabel = contentView.viewWithTag(2005) as? NSTextField {
                let attr = NSMutableAttributedString(string: "● ", attributes: [.foregroundColor: gpuInfo.pressure.color])
                attr.append(NSAttributedString(string: String(format: "GPU: %.0f%%", gpuInfo.usagePercent), attributes: [
                    .foregroundColor: NSColor.lightGray,
                    .font: NSFont.systemFont(ofSize: 12)
                ]))
                gpuLabel.attributedStringValue = attr
            }

            // Schedule next update
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                self.updateSettingsPanelBars()
            }
        }
    }

    // MARK: - Local RAG Configuration

    @objc func configureRAGAction(_ sender: NSMenuItem) {
        debugLog("Configure RAG action triggered")

        // Show folder picker to select project root
        let openPanel = NSOpenPanel()
        openPanel.canChooseFiles = false
        openPanel.canChooseDirectories = true
        openPanel.allowsMultipleSelection = false
        openPanel.message = "Select a project folder to enable GraphRAG temporal indexing"
        openPanel.prompt = "Enable RAG"
        openPanel.title = "Configure GraphRAG"

        let response = openPanel.runModal()

        if response == .OK, let url = openPanel.url {
            let projectPath = url.path
            debugLog("Selected project for RAG: \(projectPath)")

            // Update the server script to include project root
            enableRAGForProject(projectPath)
        }
    }

    private func enableRAGForProject(_ projectPath: String) {
        debugLog("Enabling RAG for project: \(projectPath)")

        // Save project path to config
        let configPath = "\(NSHomeDirectory())/.claude/mageagent/rag-config.json"
        let config: [String: Any] = ["project_root": projectPath, "enabled": true]

        if let jsonData = try? JSONSerialization.data(withJSONObject: config, options: .prettyPrinted) {
            try? jsonData.write(to: URL(fileURLWithPath: configPath))
        }

        // Restart server with RAG enabled
        let alert = NSAlert()
        alert.messageText = "Enable GraphRAG"
        alert.informativeText = "GraphRAG will build a temporal knowledge graph of:\n\(projectPath)\n\nIndexes classes, functions, imports, and their relationships over time.\n\nThis requires restarting the server. Continue?"
        alert.alertStyle = .informational
        alert.addButton(withTitle: "Restart & Enable")
        alert.addButton(withTitle: "Cancel")

        if alert.runModal() == .alertFirstButtonReturn {
            // Restart server with MAGEAGENT_PROJECT_ROOT set
            restartServerWithRAG(projectPath)
        }
    }

    private func restartServerWithRAG(_ projectPath: String) {
        debugLog("Restarting server with RAG for: \(projectPath)")

        // Stop current server
        runServerScript("stop") { [weak self] _, _ in
            guard let self = self else { return }

            // Start with project root env var
            DispatchQueue.global(qos: .userInitiated).async {
                let task = Process()
                task.executableURL = URL(fileURLWithPath: "/bin/bash")

                // Start server with MAGEAGENT_PROJECT_ROOT
                let script = """
                export MAGEAGENT_PROJECT_ROOT="\(projectPath)"
                cd ~/.claude/mageagent
                nohup python3 -u server.py > ~/.claude/debug/mageagent.log 2> ~/.claude/debug/mageagent.error.log &
                echo $! > ~/.claude/mageagent/mageagent.pid
                """
                task.arguments = ["-l", "-c", script]

                var environment = ProcessInfo.processInfo.environment
                environment["HOME"] = NSHomeDirectory()
                environment["MAGEAGENT_PROJECT_ROOT"] = projectPath
                task.environment = environment

                do {
                    try task.run()
                    task.waitUntilExit()

                    DispatchQueue.main.async {
                        // Wait a moment for server to start
                        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                            self.checkServerStatus()
                            self.sendNotification(title: "Nexus Local Compute", body: "GraphRAG enabled for project")
                        }
                    }
                } catch {
                    self.debugLog("Failed to restart with RAG: \(error)")
                    DispatchQueue.main.async {
                        self.sendNotification(title: "Nexus Local Compute", body: "Failed to enable RAG: \(error.localizedDescription)")
                    }
                }
            }
        }
    }

    // MARK: - Offline Mode Toggle

    @objc func toggleOfflineModeAction(_ sender: NSMenuItem) {
        debugLog("Toggle Offline Mode action triggered")

        let command = isOfflineMode ? "disable" : "enable"
        runOfflineModeScript(command) { [weak self] success, output in
            guard let self = self else { return }

            if success {
                self.isOfflineMode = !self.isOfflineMode

                // Update menu item
                DispatchQueue.main.async {
                    sender.state = self.isOfflineMode ? .on : .off
                    sender.title = self.isOfflineMode ? "Offline Mode (Active)" : "Offline Mode"
                }

                // Notify user
                let message = self.isOfflineMode
                    ? "Offline mode enabled - Nexus Memory hooks disabled"
                    : "Online mode restored - Nexus Memory hooks re-enabled"
                self.sendNotification(title: "Nexus Local Compute", body: message)
            } else {
                self.sendNotification(title: "Nexus Local Compute", body: "Failed to toggle offline mode: \(output)")
            }
        }
    }

    private func runOfflineModeScript(_ command: String, completion: @escaping (Bool, String) -> Void) {
        debugLog("Executing offline mode script: \(command)")

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else {
                DispatchQueue.main.async { completion(false, "AppDelegate deallocated") }
                return
            }

            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/bin/bash")
            task.arguments = ["-l", "-c", "\(Config.offlineModeScript) \(command)"]

            var environment = ProcessInfo.processInfo.environment
            environment["HOME"] = NSHomeDirectory()
            task.environment = environment

            let outputPipe = Pipe()
            task.standardOutput = outputPipe
            task.standardError = outputPipe

            do {
                try task.run()
                task.waitUntilExit()

                let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
                let output = String(data: outputData, encoding: .utf8) ?? ""

                self.debugLog("Offline mode script '\(command)' completed with status: \(task.terminationStatus)")

                DispatchQueue.main.async {
                    completion(task.terminationStatus == 0, output)
                }
            } catch {
                self.debugLog("Offline mode script execution failed: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(false, error.localizedDescription)
                }
            }
        }
    }

    private func checkOfflineModeStatus() {
        runOfflineModeScript("status") { [weak self] success, output in
            guard let self = self else { return }

            // Parse output to determine mode - "Online mode" means NOT offline
            self.isOfflineMode = !output.contains("Online mode")

            // Update menu item
            DispatchQueue.main.async {
                if let offlineItem = self.menu.item(withTag: MenuItemTag.offlineMode.rawValue) {
                    offlineItem.state = self.isOfflineMode ? .on : .off
                    offlineItem.title = self.isOfflineMode ? "Offline Mode (Active)" : "Offline Mode"
                }
            }
        }
    }

    // MARK: - Model Router Control

    @objc func startRouterAction(_ sender: NSMenuItem) {
        debugLog("Start Router action triggered")
        runRouterScript("start") { [weak self] success, output in
            guard let self = self else { return }
            DispatchQueue.main.async {
                if success {
                    self.sendNotification(title: "Model Router", body: "Router started successfully")
                } else {
                    self.sendNotification(title: "Model Router", body: "Failed to start router")
                }
                self.checkRouterStatus()
            }
        }
    }

    @objc func stopRouterAction(_ sender: NSMenuItem) {
        debugLog("Stop Router action triggered")
        runRouterScript("stop") { [weak self] success, output in
            guard let self = self else { return }
            DispatchQueue.main.async {
                if success {
                    self.sendNotification(title: "Model Router", body: "Router stopped")
                }
                self.checkRouterStatus()
            }
        }
    }

    @objc func switchToOpusAction(_ sender: NSMenuItem) {
        switchRouterModel(provider: "anthropic", model: "claude-opus-4-5-20251101")
    }

    @objc func switchToSonnetAction(_ sender: NSMenuItem) {
        switchRouterModel(provider: "anthropic", model: "claude-sonnet-4-5-20250929")
    }

    @objc func switchToMageHybridAction(_ sender: NSMenuItem) {
        switchRouterModel(provider: "mageagent", model: "mageagent:hybrid")
    }

    @objc func switchToMageExecuteAction(_ sender: NSMenuItem) {
        switchRouterModel(provider: "mageagent", model: "mageagent:execute")
    }

    @objc func switchToMageFastAction(_ sender: NSMenuItem) {
        switchRouterModel(provider: "mageagent", model: "mageagent:fast")
    }

    private func switchRouterModel(provider: String, model: String) {
        debugLog("Switching router to \(provider)/\(model)")

        // First ensure router is running
        if !isRouterRunning {
            runRouterScript("start") { [weak self] success, _ in
                guard let self = self, success else {
                    self?.sendNotification(title: "Model Router", body: "Failed to start router")
                    return
                }
                // Wait a bit then switch
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    self.performRouterSwitch(provider: provider, model: model)
                }
            }
        } else {
            performRouterSwitch(provider: provider, model: model)
        }
    }

    private func performRouterSwitch(provider: String, model: String) {
        runRouterScript("switch \(provider) \(model)") { [weak self] success, output in
            guard let self = self else { return }
            DispatchQueue.main.async {
                if success {
                    self.currentRouterProvider = provider
                    self.currentRouterModel = model
                    self.updateRouterMenuItem()

                    let displayName = provider == "mageagent" ? "Local AI (\(model.replacingOccurrences(of: "mageagent:", with: "")))" : model
                    self.sendNotification(title: "Model Router", body: "Switched to \(displayName)")
                } else {
                    self.sendNotification(title: "Model Router", body: "Failed to switch model")
                }
            }
        }
    }

    private func runRouterScript(_ command: String, completion: @escaping (Bool, String) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/bin/bash")
            task.arguments = ["-l", "-c", "\(Config.modelRouterScript) \(command)"]

            let pipe = Pipe()
            task.standardOutput = pipe
            task.standardError = pipe

            do {
                try task.run()
                task.waitUntilExit()

                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let output = String(data: data, encoding: .utf8) ?? ""
                let success = task.terminationStatus == 0

                DispatchQueue.main.async {
                    completion(success, output)
                }
            } catch {
                DispatchQueue.main.async {
                    completion(false, error.localizedDescription)
                }
            }
        }
    }

    private func checkRouterStatus() {
        // Check if router is running by trying to connect
        guard let url = URL(string: "\(Config.routerURL)/router/status") else { return }

        var request = URLRequest(url: url)
        request.timeoutInterval = 2.0

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            guard let self = self else { return }

            DispatchQueue.main.async {
                if let httpResponse = response as? HTTPURLResponse,
                   httpResponse.statusCode == 200,
                   let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    self.isRouterRunning = true
                    self.currentRouterProvider = json["current_provider"] as? String ?? "anthropic"
                    self.currentRouterModel = json["current_model"] as? String ?? "unknown"
                } else {
                    self.isRouterRunning = false
                }
                self.updateRouterMenuItem()
            }
        }.resume()
    }

    private func updateRouterMenuItem() {
        // Find router submenu and update status
        if let routerItem = menu.item(withTag: MenuItemTag.modelRouter.rawValue),
           let submenu = routerItem.submenu,
           let statusItem = submenu.item(withTag: MenuItemTag.routerStatus.rawValue) {

            if isRouterRunning {
                let shortModel = currentRouterModel.replacingOccurrences(of: "claude-", with: "").replacingOccurrences(of: "mageagent:", with: "")
                statusItem.title = "● \(currentRouterProvider)/\(shortModel)"

                // Color the indicator
                let attr = NSMutableAttributedString(string: "● ", attributes: [.foregroundColor: NSColor.systemGreen])
                attr.append(NSAttributedString(string: "\(currentRouterProvider)/\(shortModel)"))
                statusItem.attributedTitle = attr
            } else {
                let attr = NSMutableAttributedString(string: "● ", attributes: [.foregroundColor: NSColor.systemRed])
                attr.append(NSAttributedString(string: "Router: Stopped"))
                statusItem.attributedTitle = attr
            }

            // Update checkmarks on model items
            for item in submenu.items {
                switch item.tag {
                case MenuItemTag.switchToOpus.rawValue:
                    item.state = (currentRouterProvider == "anthropic" && currentRouterModel.contains("opus")) ? .on : .off
                case MenuItemTag.switchToSonnet.rawValue:
                    item.state = (currentRouterProvider == "anthropic" && currentRouterModel.contains("sonnet")) ? .on : .off
                case MenuItemTag.switchToMageHybrid.rawValue:
                    item.state = (currentRouterProvider == "mageagent" && currentRouterModel.contains("hybrid")) ? .on : .off
                case MenuItemTag.switchToMageExecute.rawValue:
                    item.state = (currentRouterProvider == "mageagent" && currentRouterModel.contains("execute")) ? .on : .off
                case MenuItemTag.switchToMageFast.rawValue:
                    item.state = (currentRouterProvider == "mageagent" && currentRouterModel.contains("fast")) ? .on : .off
                default:
                    break
                }
            }
        }
    }

    // MARK: - Server Status Checking

    private func startStatusTimer() {
        statusTimer?.invalidate()
        statusTimer = Timer.scheduledTimer(withTimeInterval: Config.statusCheckInterval, repeats: true) { [weak self] _ in
            self?.checkServerStatus()
            self?.checkRouterStatus()
        }
        // Add to common run loop mode to ensure it fires even during menu tracking
        RunLoop.current.add(statusTimer!, forMode: .common)

        // Also check router status immediately
        checkRouterStatus()
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
                        let ragStatus = json["rag"] as? [String: Any]
                        self.updateServerStatus(running: true, version: version, models: models, rag: ragStatus)
                    }
                } catch {
                    self.debugLog("Failed to parse health response: \(error)")
                    self.updateServerStatus(running: false, version: nil, models: [])
                }
            }
        }.resume()
    }

    private func updateServerStatus(running: Bool, version: String?, models: [String], rag: [String: Any]? = nil) {
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

        // Update RAG status menu item
        if let ragItem = menu.item(withTag: MenuItemTag.ragStatus.rawValue) {
            if let ragInfo = rag {
                let ragType = ragInfo["type"] as? String ?? "rag"
                let entities = ragInfo["total_entities"] as? Int ?? ragInfo["total_patterns"] as? Int ?? 0
                let relationships = ragInfo["total_relationships"] as? Int ?? 0
                let language = ragInfo["primary_language"] as? String ?? "?"

                if ragType == "graphrag" {
                    ragItem.title = "✓ GraphRAG: \(entities) entities, \(relationships) rels (\(language))"
                    ragItem.toolTip = "Temporal Knowledge Graph active. Click to change project."
                } else {
                    ragItem.title = "✓ Local RAG: \(entities) patterns (\(language))"
                    ragItem.toolTip = "RAG is active. Click to change project."
                }
            } else if running {
                ragItem.title = "GraphRAG: Click to Enable"
                ragItem.toolTip = "Click to select a project folder for GraphRAG indexing"
            } else {
                ragItem.title = "GraphRAG: Server Offline"
                ragItem.toolTip = "Start the server first to enable GraphRAG"
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
        } else if let symbolImage = NSImage(systemSymbolName: "brain.head.profile", accessibilityDescription: "Nexus Local Compute") {
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

        // Also fetch throughput stats from server (separate async call)
        fetchThroughputStats()
    }

    /// Fetch throughput statistics from the local AI server
    private func fetchThroughputStats() {
        guard isServerRunning else {
            lastThroughput = (0, 0, "")
            return
        }

        guard let url = URL(string: "\(Config.mageagentURL)/stats") else { return }

        var request = URLRequest(url: url)
        request.timeoutInterval = 2.0

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            guard let self = self,
                  let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200,
                  let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return
            }

            DispatchQueue.main.async {
                let tokensPerSec = json["last_tokens_per_sec"] as? Double ?? 0
                let totalTokens = json["total_tokens_generated"] as? Int ?? 0
                let lastModel = json["last_model"] as? String ?? ""

                // Only update if there was recent activity (within last 30 seconds)
                if let lastInference = json["last_inference"] as? Double {
                    let now = Date().timeIntervalSince1970
                    if now - lastInference < 30 {
                        self.lastThroughput = (tokensPerSec, totalTokens, lastModel)
                    } else {
                        self.lastThroughput = (0, totalTokens, lastModel)  // Idle but show total
                    }
                } else {
                    self.lastThroughput = (0, 0, "")
                }
            }
        }.resume()
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

        // Calculate memory usage to match Activity Monitor's "Memory Used"
        // Activity Monitor: Memory Used = App Memory + Wired Memory + Compressed
        // Where App Memory ≈ (Active + Inactive + Speculative + Throttled + Purgeable that's in use)
        // Simplified: Used = Total - Free - Inactive - Purgeable - Speculative + Compressed
        // Or more directly: Used = Active + Inactive + Speculative + Wired + Compressed - Purgeable

        let active = UInt64(stats.active_count) * pageSize
        let inactive = UInt64(stats.inactive_count) * pageSize
        let speculative = UInt64(stats.speculative_count) * pageSize
        let wired = UInt64(stats.wire_count) * pageSize
        let compressed = UInt64(stats.compressor_page_count) * pageSize
        let purgeable = UInt64(stats.purgeable_count) * pageSize
        let external = UInt64(stats.external_page_count) * pageSize

        // Match Activity Monitor's calculation:
        // App Memory = Active + Inactive + Speculative + Throttled - Purgeable - External (file cache)
        // Memory Used = App Memory + Wired + Compressed
        let appMemory = active + inactive + speculative - purgeable - external
        let used = appMemory + wired + compressed

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

    /// Get GPU/Metal information with estimated VRAM usage
    private func getGPUInfo() -> (description: String, pressure: SystemPressure, usagePercent: Double) {
        // For Apple Silicon, GPU shares unified memory with CPU
        // We estimate GPU memory usage based on loaded model sizes

        // Model sizes in GB (approximate)
        let modelSizes: [String: Double] = [
            "mageagent:primary": 77.0,    // Qwen-72B Q8
            "mageagent:competitor": 18.0, // Qwen-32B Q4
            "mageagent:tools": 9.0,       // Hermes-3 8B Q8
            "mageagent:validator": 5.0,   // Qwen-7B Q4
            "mageagent:glm": 17.0         // GLM-4.7 Flash 4bit
        ]

        // Get total unified memory
        var totalMemory: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        sysctlbyname("hw.memsize", &totalMemory, &size, nil, 0)
        let totalGB = Double(totalMemory) / (1024 * 1024 * 1024)

        // Calculate estimated GPU memory usage from loaded models
        var gpuUsedGB: Double = 0
        for model in loadedModels {
            gpuUsedGB += modelSizes[model] ?? 0
        }

        let usagePercent = (gpuUsedGB / totalGB) * 100

        // Determine pressure based on GPU memory usage
        var pressure: SystemPressure = .nominal
        var description = "GPU: Idle"

        if isServerRunning {
            if gpuUsedGB > 0 {
                // Format: "GPU: XX.X% (XXX GB)"
                description = String(format: "GPU: %.0f%% (%.0f GB)", usagePercent, gpuUsedGB)

                // Pressure thresholds for GPU memory
                if usagePercent >= 85 {
                    pressure = .critical
                } else if usagePercent >= 60 {
                    pressure = .warning
                }
            } else {
                description = "GPU: 0% (Standby)"
            }
        }

        return (description, pressure, usagePercent)
    }

    /// Update the menu items with current pressure information
    private func updatePressureMenuItems(memory: (usedGB: Double, totalGB: Double, pressure: SystemPressure, percentUsed: Double),
                                         cpu: (usage: Double, pressure: SystemPressure),
                                         gpu: (description: String, pressure: SystemPressure, usagePercent: Double)) {
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

        // Update memory pressure bar graph in menu
        updateMenuMemoryPressureBar()

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

        // Update Neural Engine TOPS - always green (informational)
        if let neItem = menu.item(withTag: MenuItemTag.neuralEngineDetail.rawValue) {
            let neText: String
            if neuralEngineTOPS > 0 {
                neText = String(format: "Neural Engine: %.0f TOPS", neuralEngineTOPS)
            } else {
                neText = "Neural Engine: N/A"
            }
            let neAttr = NSMutableAttributedString(string: "  ● ", attributes: [.foregroundColor: NSColor.systemBlue])
            neAttr.append(NSAttributedString(string: neText, attributes: [
                .foregroundColor: textColor,
                .font: NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .semibold)
            ]))
            neItem.attributedTitle = neAttr
        }

        // Update Throughput display
        if let throughputItem = menu.item(withTag: MenuItemTag.throughputDetail.rawValue) {
            let throughputText: String
            let throughputColor: NSColor
            if lastThroughput.tokensPerSec > 0 {
                // Active inference - show tok/s and total
                if lastThroughput.totalTokens > 0 {
                    throughputText = String(format: "Tokens: %.1f tok/s (%d total)", lastThroughput.tokensPerSec, lastThroughput.totalTokens)
                } else {
                    throughputText = String(format: "Tokens: %.1f tok/s", lastThroughput.tokensPerSec)
                }
                throughputColor = NSColor.systemGreen
            } else if isServerRunning {
                // Idle but server running - show total tokens if any
                if lastThroughput.totalTokens > 0 {
                    throughputText = String(format: "Tokens: Idle (%d total)", lastThroughput.totalTokens)
                } else {
                    throughputText = "Tokens: Idle"
                }
                throughputColor = NSColor.systemGray
            } else {
                throughputText = "Tokens: Server Offline"
                throughputColor = NSColor.systemGray
            }
            let throughputAttr = NSMutableAttributedString(string: "  ● ", attributes: [.foregroundColor: throughputColor])
            throughputAttr.append(NSAttributedString(string: throughputText, attributes: [
                .foregroundColor: textColor,
                .font: NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .semibold)
            ]))
            throughputItem.attributedTitle = throughputAttr
        }

        // Store current values
        memoryPressure = memory.pressure
        cpuUsage = cpu.usage
    }

    // MARK: - Chat Interface

    @objc func openChatAction(_ sender: NSMenuItem) {
        debugLog("Open Chat action triggered")

        // Force recreate if attachment views are missing (old window version)
        if chatWindow != nil && attachmentStackView == nil {
            debugLog("Recreating chat window - missing attachment support")
            chatWindow?.close()
            chatWindow = nil
        }

        if chatWindow == nil {
            createChatWindow()
        }

        chatWindow?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        chatInputField?.becomeFirstResponder()
    }

    private func createChatWindow() {
        let windowWidth: CGFloat = 500
        let windowHeight: CGFloat = 600

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: windowWidth, height: windowHeight),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )

        window.title = "Nexus Local Compute - Chat"
        window.center()
        window.isReleasedWhenClosed = false
        window.minSize = NSSize(width: 400, height: 400)
        window.backgroundColor = NSColor(calibratedWhite: 0.12, alpha: 1.0)

        let contentView = NSView(frame: NSRect(x: 0, y: 0, width: windowWidth, height: windowHeight))
        contentView.wantsLayer = true
        contentView.layer?.backgroundColor = NSColor(calibratedWhite: 0.12, alpha: 1.0).cgColor

        // Header with model info
        let headerHeight: CGFloat = 40
        let headerView = NSView(frame: NSRect(x: 0, y: windowHeight - headerHeight, width: windowWidth, height: headerHeight))
        headerView.wantsLayer = true
        headerView.layer?.backgroundColor = NSColor(calibratedWhite: 0.15, alpha: 1.0).cgColor
        headerView.autoresizingMask = [.width, .minYMargin]

        let modelLabel = NSTextField(labelWithString: "Local AI - Hybrid Mode")
        modelLabel.frame = NSRect(x: 15, y: 10, width: windowWidth - 100, height: 20)
        modelLabel.font = NSFont.systemFont(ofSize: 14, weight: .semibold)
        modelLabel.textColor = .white
        modelLabel.autoresizingMask = [.width]
        modelLabel.identifier = NSUserInterfaceItemIdentifier("chatModelLabel")
        headerView.addSubview(modelLabel)

        let clearButton = NSButton(title: "Clear", target: self, action: #selector(clearChatAction(_:)))
        clearButton.frame = NSRect(x: windowWidth - 70, y: 8, width: 55, height: 24)
        clearButton.bezelStyle = .rounded
        clearButton.autoresizingMask = [.minXMargin]
        headerView.addSubview(clearButton)

        contentView.addSubview(headerView)

        // Chat output area (scrollable text view)
        let inputHeight: CGFloat = 60
        let outputHeight = windowHeight - headerHeight - inputHeight - 20

        let scrollView = NSScrollView(frame: NSRect(x: 10, y: inputHeight + 10, width: windowWidth - 20, height: outputHeight))
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autoresizingMask = [.width, .height]
        scrollView.borderType = .noBorder
        scrollView.drawsBackground = true
        scrollView.backgroundColor = NSColor(calibratedWhite: 0.08, alpha: 1.0)
        scrollView.wantsLayer = true
        scrollView.layer?.cornerRadius = 8

        let textView = NSTextView(frame: scrollView.bounds)
        textView.isEditable = false
        textView.isSelectable = true
        textView.font = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
        textView.backgroundColor = NSColor(calibratedWhite: 0.08, alpha: 1.0)
        textView.textColor = .white
        textView.autoresizingMask = [.width]
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.containerSize = NSSize(width: scrollView.contentSize.width, height: CGFloat.greatestFiniteMagnitude)
        textView.textContainerInset = NSSize(width: 10, height: 10)

        // Add welcome message
        let welcomeMessage = "Welcome to Nexus Local Compute!\n\nType a message below to chat with your local AI models.\nUsing: mageagent:hybrid (Qwen-72B + Hermes-3)\n\n"
        textView.string = welcomeMessage

        scrollView.documentView = textView
        contentView.addSubview(scrollView)
        chatOutputTextView = textView

        // Input area with attachment support
        let inputContainerHeight: CGFloat = inputHeight - 10
        let inputContainer = NSView(frame: NSRect(x: 10, y: 10, width: windowWidth - 20, height: inputContainerHeight))
        inputContainer.wantsLayer = true
        inputContainer.layer?.backgroundColor = NSColor(calibratedWhite: 0.15, alpha: 1.0).cgColor
        inputContainer.layer?.cornerRadius = 8
        inputContainer.autoresizingMask = [.width, .maxYMargin]
        inputContainer.identifier = NSUserInterfaceItemIdentifier("chatInputContainer")

        // Attachment display area (shown above input when files attached)
        let attachContainer = NSView(frame: NSRect(x: 5, y: 48, width: windowWidth - 30, height: 30))
        attachContainer.wantsLayer = true
        attachContainer.layer?.backgroundColor = NSColor(calibratedWhite: 0.2, alpha: 1.0).cgColor
        attachContainer.layer?.cornerRadius = 6
        attachContainer.autoresizingMask = [.width]
        attachContainer.isHidden = true
        attachmentContainerView = attachContainer

        let attachStack = NSStackView(frame: NSRect(x: 5, y: 3, width: windowWidth - 40, height: 24))
        attachStack.orientation = .horizontal
        attachStack.spacing = 8
        attachStack.alignment = .centerY
        attachStack.distribution = .fillProportionally
        attachStack.autoresizingMask = [.width]
        attachmentStackView = attachStack
        attachContainer.addSubview(attachStack)
        inputContainer.addSubview(attachContainer)

        // Input row with attach button, text field, and send button
        let attachButton = NSButton(frame: NSRect(x: 8, y: 10, width: 28, height: 30))
        attachButton.bezelStyle = .rounded
        attachButton.title = "+"
        attachButton.font = NSFont.systemFont(ofSize: 16, weight: .medium)
        attachButton.target = self
        attachButton.action = #selector(attachFileAction(_:))
        attachButton.toolTip = "Attach files (⌘O)"
        attachButton.keyEquivalent = "o"
        attachButton.keyEquivalentModifierMask = [.command]
        inputContainer.addSubview(attachButton)

        let inputField = NSTextField(frame: NSRect(x: 40, y: 10, width: windowWidth - 130, height: 30))
        inputField.placeholderString = "Type your message..."
        inputField.font = NSFont.systemFont(ofSize: 14)
        inputField.backgroundColor = NSColor(calibratedWhite: 0.2, alpha: 1.0)
        inputField.textColor = .white
        inputField.isBordered = true
        inputField.bezelStyle = .roundedBezel
        inputField.focusRingType = .none
        inputField.autoresizingMask = [.width]
        inputField.target = self
        inputField.action = #selector(sendChatMessage(_:))
        inputContainer.addSubview(inputField)
        chatInputField = inputField

        let sendButton = NSButton(title: "Send", target: self, action: #selector(sendChatMessage(_:)))
        sendButton.frame = NSRect(x: windowWidth - 80, y: 10, width: 55, height: 30)
        sendButton.bezelStyle = .rounded
        sendButton.autoresizingMask = [.minXMargin]
        inputContainer.addSubview(sendButton)

        contentView.addSubview(inputContainer)

        window.contentView = contentView
        chatWindow = window

        // Update model label based on router state
        updateChatModelLabel()
    }

    private func updateChatModelLabel() {
        // Chat always uses mageagent:hybrid directly, not through router
        // So label always shows "Local AI - Hybrid Mode"
        // No need to update since it's set correctly at creation
    }

    @objc func clearChatAction(_ sender: NSButton) {
        chatMessages.removeAll()
        attachedFiles.removeAll()
        updateAttachmentDisplay()
        chatOutputTextView?.string = "Chat cleared.\n\n"
    }

    // MARK: - Usage Analytics Dashboard

    @objc func openAnalyticsAction(_ sender: NSMenuItem) {
        debugLog("Open Analytics action triggered")

        if analyticsWindow == nil {
            createAnalyticsWindow()
        }

        // Refresh data when opening
        refreshAnalytics()

        analyticsWindow?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func createAnalyticsWindow() {
        let windowWidth: CGFloat = 700
        let windowHeight: CGFloat = 500

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: windowWidth, height: windowHeight),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )

        window.title = "Nexus Local Compute - Usage Analytics"
        window.center()
        window.isReleasedWhenClosed = false
        window.minSize = NSSize(width: 600, height: 400)
        window.backgroundColor = NSColor(calibratedWhite: 0.12, alpha: 1.0)

        let contentView = NSView(frame: NSRect(x: 0, y: 0, width: windowWidth, height: windowHeight))
        contentView.wantsLayer = true
        contentView.layer?.backgroundColor = NSColor(calibratedWhite: 0.12, alpha: 1.0).cgColor

        // Header
        let headerHeight: CGFloat = 50
        let headerView = NSView(frame: NSRect(x: 0, y: windowHeight - headerHeight, width: windowWidth, height: headerHeight))
        headerView.wantsLayer = true
        headerView.layer?.backgroundColor = NSColor(calibratedWhite: 0.15, alpha: 1.0).cgColor
        headerView.autoresizingMask = [.width, .minYMargin]

        let titleLabel = NSTextField(labelWithString: "📊 Token Usage & Analytics")
        titleLabel.frame = NSRect(x: 15, y: 15, width: windowWidth - 160, height: 24)
        titleLabel.font = NSFont.systemFont(ofSize: 18, weight: .bold)
        titleLabel.textColor = .white
        titleLabel.autoresizingMask = [.width]
        headerView.addSubview(titleLabel)

        let refreshButton = NSButton(title: "Refresh", target: self, action: #selector(refreshAnalytics))
        refreshButton.frame = NSRect(x: windowWidth - 140, y: 13, width: 70, height: 28)
        refreshButton.bezelStyle = .rounded
        refreshButton.autoresizingMask = [.minXMargin]
        headerView.addSubview(refreshButton)

        let resetButton = NSButton(title: "Reset", target: self, action: #selector(resetAnalytics))
        resetButton.frame = NSRect(x: windowWidth - 65, y: 13, width: 55, height: 28)
        resetButton.bezelStyle = .rounded
        resetButton.autoresizingMask = [.minXMargin]
        headerView.addSubview(resetButton)

        contentView.addSubview(headerView)

        // Analytics output area (scrollable text view)
        let scrollView = NSScrollView(frame: NSRect(x: 15, y: 15, width: windowWidth - 30, height: windowHeight - headerHeight - 25))
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autoresizingMask = [.width, .height]
        scrollView.borderType = .noBorder
        scrollView.drawsBackground = true
        scrollView.backgroundColor = NSColor(calibratedWhite: 0.08, alpha: 1.0)
        scrollView.wantsLayer = true
        scrollView.layer?.cornerRadius = 8

        let textView = NSTextView(frame: scrollView.bounds)
        textView.isEditable = false
        textView.isSelectable = true
        textView.font = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
        textView.backgroundColor = NSColor(calibratedWhite: 0.08, alpha: 1.0)
        textView.textColor = .white
        textView.autoresizingMask = [.width]
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.containerSize = NSSize(width: scrollView.contentSize.width, height: CGFloat.greatestFiniteMagnitude)
        textView.textContainerInset = NSSize(width: 15, height: 15)

        textView.string = "Loading analytics...\n\n"

        scrollView.documentView = textView
        contentView.addSubview(scrollView)

        window.contentView = contentView
        analyticsWindow = window
        analyticsTextView = textView
    }

    @objc func refreshAnalytics() {
        debugLog("Refreshing analytics data")

        guard let url = URL(string: "\(Config.mageagentURL)/stats") else {
            analyticsTextView?.string = "Error: Invalid server URL\n"
            return
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 10

        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            guard let self = self else { return }

            if let error = error {
                DispatchQueue.main.async {
                    self.analyticsTextView?.string = "Error fetching stats: \(error.localizedDescription)\n"
                }
                return
            }

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200,
                  let data = data else {
                DispatchQueue.main.async {
                    self.analyticsTextView?.string = "Error: Invalid server response\n"
                }
                return
            }

            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    let formattedOutput = self.formatAnalyticsData(json)
                    DispatchQueue.main.async {
                        self.analyticsTextView?.string = formattedOutput
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.analyticsTextView?.string = "Error parsing stats: \(error.localizedDescription)\n"
                }
            }
        }.resume()
    }

    private func formatAnalyticsData(_ json: [String: Any]) -> String {
        var output = ""

        // Session info
        if let session = json["session"] as? [String: Any] {
            output += "═══════════════════════════════════════════════════════════\n"
            output += "  SESSION OVERVIEW\n"
            output += "═══════════════════════════════════════════════════════════\n\n"

            if let duration = session["duration_human"] as? String {
                output += "Session Duration: \(duration)\n"
            }
            output += "\n"
        }

        // Totals
        if let totals = json["totals"] as? [String: Any] {
            output += "───────────────────────────────────────────────────────────\n"
            output += "  TOTAL USAGE\n"
            output += "───────────────────────────────────────────────────────────\n\n"

            if let requests = totals["requests"] as? Int {
                output += "Total Requests:      \(requests)\n"
            }
            if let promptTokens = totals["prompt_tokens"] as? Int {
                output += "Prompt Tokens:       \(promptTokens.formatted())\n"
            }
            if let completionTokens = totals["completion_tokens"] as? Int {
                output += "Completion Tokens:   \(completionTokens.formatted())\n"
            }
            if let totalTokens = totals["total_tokens"] as? Int {
                output += "Total Tokens:        \(totalTokens.formatted())\n"
            }
            output += "\n"
        }

        // Last request
        if let last = json["last_request"] as? [String: Any] {
            output += "───────────────────────────────────────────────────────────\n"
            output += "  LAST REQUEST\n"
            output += "───────────────────────────────────────────────────────────\n\n"

            if let model = last["model"] as? String {
                output += "Model:               \(model)\n"
            }
            if let pattern = last["pattern"] as? String {
                output += "Pattern:             \(pattern)\n"
            }
            if let tokensPerSec = last["tokens_per_sec"] as? Double {
                output += "Speed:               \(String(format: "%.1f", tokensPerSec)) tok/s\n"
            }
            if let tokens = last["tokens_generated"] as? Int {
                output += "Tokens Generated:    \(tokens)\n"
            }
            if let duration = last["duration_sec"] as? Double {
                output += "Duration:            \(String(format: "%.2f", duration))s\n"
            }
            output += "\n"
        }

        // By model
        if let byModel = json["by_model"] as? [String: Any], !byModel.isEmpty {
            output += "───────────────────────────────────────────────────────────\n"
            output += "  USAGE BY MODEL\n"
            output += "───────────────────────────────────────────────────────────\n\n"

            let sortedModels = byModel.keys.sorted()
            for modelKey in sortedModels {
                if let modelData = byModel[modelKey] as? [String: Any] {
                    let requests = modelData["requests"] as? Int ?? 0
                    let promptTokens = modelData["prompt_tokens"] as? Int ?? 0
                    let completionTokens = modelData["completion_tokens"] as? Int ?? 0

                    output += "Model: \(modelKey)\n"
                    output += "  Requests:          \(requests)\n"
                    output += "  Prompt Tokens:     \(promptTokens.formatted())\n"
                    output += "  Completion Tokens: \(completionTokens.formatted())\n"
                    output += "  Total Tokens:      \((promptTokens + completionTokens).formatted())\n\n"
                }
            }
        }

        // By pattern
        if let byPattern = json["by_pattern"] as? [String: Any], !byPattern.isEmpty {
            output += "───────────────────────────────────────────────────────────\n"
            output += "  USAGE BY PATTERN\n"
            output += "───────────────────────────────────────────────────────────\n\n"

            let sortedPatterns = byPattern.keys.sorted()
            for patternKey in sortedPatterns {
                if let patternData = byPattern[patternKey] as? [String: Any] {
                    let requests = patternData["requests"] as? Int ?? 0
                    let tokens = patternData["tokens"] as? Int ?? 0

                    output += "Pattern: \(patternKey)\n"
                    output += "  Requests:     \(requests)\n"
                    output += "  Tokens:       \(tokens.formatted())\n\n"
                }
            }
        }

        output += "═══════════════════════════════════════════════════════════\n"
        output += "Last Updated: \(Date().formatted(date: .omitted, time: .standard))\n"
        output += "═══════════════════════════════════════════════════════════\n"

        return output
    }

    @objc func resetAnalytics() {
        let alert = NSAlert()
        alert.messageText = "Reset Analytics?"
        alert.informativeText = "This will clear all usage statistics and token counts. This action cannot be undone."
        alert.addButton(withTitle: "Reset")
        alert.addButton(withTitle: "Cancel")
        alert.alertStyle = .warning

        let response = alert.runModal()
        if response == .alertFirstButtonReturn {
            debugLog("Resetting analytics")

            guard let url = URL(string: "\(Config.mageagentURL)/stats/reset") else { return }

            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.timeoutInterval = 10

            URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
                guard let self = self else { return }

                if let error = error {
                    DispatchQueue.main.async {
                        self.sendNotification(title: "Nexus Local Compute", body: "Failed to reset analytics: \(error.localizedDescription)")
                    }
                    return
                }

                guard let httpResponse = response as? HTTPURLResponse,
                      httpResponse.statusCode == 200 else {
                    DispatchQueue.main.async {
                        self.sendNotification(title: "Nexus Local Compute", body: "Failed to reset analytics")
                    }
                    return
                }

                DispatchQueue.main.async {
                    self.sendNotification(title: "Nexus Local Compute", body: "Analytics reset successfully")
                    self.refreshAnalytics()
                }
            }.resume()
        }
    }

    // MARK: - File Attachment Methods

    @objc func attachFileAction(_ sender: Any) {
        debugLog("Attach file action triggered")

        let openPanel = NSOpenPanel()
        openPanel.canChooseFiles = true
        openPanel.canChooseDirectories = false
        openPanel.allowsMultipleSelection = true
        openPanel.treatsFilePackagesAsDirectories = false
        // Allow all file types by not setting allowedContentTypes
        openPanel.message = "Select files to attach to your message"
        openPanel.prompt = "Attach"
        openPanel.title = "Attach Files"

        // Use runModal for reliability instead of beginSheetModal
        let response = openPanel.runModal()

        if response == .OK {
            debugLog("Files selected: \(openPanel.urls.count)")
            for url in openPanel.urls {
                // Check if file is already attached
                if !self.attachedFiles.contains(where: { $0.url == url }) {
                    self.attachedFiles.append((url: url, name: url.lastPathComponent))
                    debugLog("Attached: \(url.lastPathComponent)")
                }
            }
            self.updateAttachmentDisplay()
        } else {
            debugLog("File selection cancelled")
        }
    }

    private func updateAttachmentDisplay() {
        debugLog("updateAttachmentDisplay called, files count: \(attachedFiles.count)")

        guard let container = attachmentContainerView else {
            debugLog("ERROR: container is nil")
            return
        }

        debugLog("Container is valid, updating display")

        // Clear existing subviews
        container.subviews.forEach { $0.removeFromSuperview() }

        if attachedFiles.isEmpty {
            debugLog("No files attached, hiding container")
            container.isHidden = true
            updateInputContainerHeight(hasAttachments: false)
        } else {
            debugLog("Showing \(attachedFiles.count) attached files")
            container.isHidden = false

            // Layout chips horizontally
            var xOffset: CGFloat = 8
            for (index, file) in attachedFiles.enumerated() {
                let chip = createAttachmentChip(name: file.name, index: index)
                chip.frame.origin = CGPoint(x: xOffset, y: 3)
                container.addSubview(chip)
                xOffset += chip.frame.width + 8
                debugLog("Added chip for: \(file.name) at x=\(chip.frame.origin.x)")
            }

            updateInputContainerHeight(hasAttachments: true)
            container.needsDisplay = true
        }
    }

    private func createAttachmentChip(name: String, index: Int) -> NSView {
        let chip = NSView(frame: NSRect(x: 0, y: 0, width: 150, height: 24))
        chip.wantsLayer = true
        chip.layer?.backgroundColor = NSColor(calibratedWhite: 0.25, alpha: 1.0).cgColor
        chip.layer?.cornerRadius = 12

        // File icon
        let icon = NSImageView(frame: NSRect(x: 6, y: 4, width: 16, height: 16))
        icon.image = NSImage(systemSymbolName: "doc.fill", accessibilityDescription: "File")
        icon.contentTintColor = .white
        chip.addSubview(icon)

        // Truncated file name
        let displayName = name.count > 15 ? String(name.prefix(12)) + "..." : name
        let label = NSTextField(labelWithString: displayName)
        label.frame = NSRect(x: 24, y: 3, width: 100, height: 18)
        label.font = NSFont.systemFont(ofSize: 11)
        label.textColor = .white
        label.lineBreakMode = .byTruncatingMiddle
        label.toolTip = name
        chip.addSubview(label)

        // Remove button
        let removeButton = NSButton(frame: NSRect(x: 126, y: 4, width: 18, height: 16))
        removeButton.bezelStyle = .inline
        removeButton.image = NSImage(systemSymbolName: "xmark.circle.fill", accessibilityDescription: "Remove")
        removeButton.imagePosition = .imageOnly
        removeButton.isBordered = false
        removeButton.tag = index
        removeButton.target = self
        removeButton.action = #selector(removeAttachment(_:))
        chip.addSubview(removeButton)

        // Set width based on content
        let width = min(150, CGFloat(displayName.count * 7) + 50)
        chip.frame.size.width = width
        removeButton.frame.origin.x = width - 22

        return chip
    }

    @objc func removeAttachment(_ sender: NSButton) {
        let index = sender.tag
        if index >= 0 && index < attachedFiles.count {
            attachedFiles.remove(at: index)
            updateAttachmentDisplay()
        }
    }

    private func updateInputContainerHeight(hasAttachments: Bool) {
        debugLog("updateInputContainerHeight: hasAttachments=\(hasAttachments)")

        guard let window = chatWindow,
              let contentView = window.contentView else {
            debugLog("ERROR: window or contentView is nil")
            return
        }

        // Find input container by identifier
        guard let inputContainer = contentView.subviews.first(where: {
            $0.identifier?.rawValue == "chatInputContainer"
        }) else {
            debugLog("ERROR: inputContainer not found")
            return
        }

        let baseHeight: CGFloat = 50
        let attachmentHeight: CGFloat = hasAttachments ? 38 : 0
        let newHeight = baseHeight + attachmentHeight

        debugLog("Resizing input container from \(inputContainer.frame.size.height) to \(newHeight)")

        inputContainer.frame.size.height = newHeight

        // Adjust scroll view height
        if let scrollView = contentView.subviews.first(where: { $0 is NSScrollView }) as? NSScrollView {
            let headerHeight: CGFloat = 40
            scrollView.frame.origin.y = newHeight + 10
            scrollView.frame.size.height = contentView.bounds.height - headerHeight - newHeight - 20
        }

        // Force layout update
        inputContainer.needsLayout = true
        inputContainer.needsDisplay = true
        contentView.needsLayout = true
    }

    private func readFileContents(_ url: URL) -> String? {
        let ext = url.pathExtension.lowercased()
        let fileName = url.lastPathComponent

        // Get file size
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let fileSize = attrs[.size] as? Int else {
            return "[Unable to read file: \(fileName)]"
        }

        let fileSizeKB = fileSize / 1024
        let maxTextSize = 50_000  // 50KB for text content in chat

        // Handle PDFs - extract text using Python
        if ext == "pdf" {
            return extractPDFText(url, fileSizeKB: fileSizeKB)
        }

        // Handle images - not supported in text chat
        if ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp", "heic"].contains(ext) {
            return """
            [IMAGE: \(fileName) (\(fileSizeKB)KB)]
            Note: Image analysis is not available in this text-based chat.
            To analyze images, use Claude Code directly with the Read tool.
            """
        }

        // Handle Office documents
        if ["doc", "docx", "xls", "xlsx", "ppt", "pptx"].contains(ext) {
            return """
            [DOCUMENT: \(fileName) (\(fileSizeKB)KB)]
            Note: Office documents cannot be read directly.
            Please export to PDF or plain text format for analysis.
            """
        }

        // Handle text-based files
        let textExtensions = ["txt", "md", "json", "xml", "yaml", "yml", "csv", "swift", "py", "js",
                              "ts", "tsx", "jsx", "html", "css", "scss", "less", "sh", "bash", "zsh",
                              "c", "cpp", "h", "hpp", "java", "kt", "go", "rs", "rb", "php", "sql",
                              "log", "conf", "ini", "toml", "env", "gitignore", "dockerfile", "makefile",
                              "r", "scala", "lua", "vim", "el", "clj", "ex", "exs", "erl", "hs"]

        if textExtensions.contains(ext) || ext.isEmpty {
            if let content = try? String(contentsOf: url, encoding: .utf8) {
                // Truncate large files with summary
                if content.count > maxTextSize {
                    let truncated = String(content.prefix(maxTextSize))
                    let lineCount = content.components(separatedBy: "\n").count
                    let truncatedLines = truncated.components(separatedBy: "\n").count
                    return """
                    [FILE: \(fileName) - TRUNCATED (\(fileSizeKB)KB, \(lineCount) lines total)]
                    Showing first \(truncatedLines) lines:

                    \(truncated)

                    ... [TRUNCATED - \(lineCount - truncatedLines) more lines]
                    """
                }
                return content
            }
        }

        // Try to read as text anyway
        if let content = try? String(contentsOf: url, encoding: .utf8) {
            if content.count > maxTextSize {
                return String(content.prefix(maxTextSize)) + "\n\n... [TRUNCATED]"
            }
            return content
        }

        return "[Unable to read file: \(fileName) (\(fileSizeKB)KB)]"
    }

    private func extractPDFText(_ url: URL, fileSizeKB: Int) -> String {
        let fileName = url.lastPathComponent

        // Use Python to extract PDF text
        let pythonScript = """
        import sys
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(sys.argv[1])
            text = ""
            for page_num, page in enumerate(doc):
                text += f"\\n--- Page {page_num + 1} ---\\n"
                text += page.get_text()
            print(text[:50000] if len(text) > 50000 else text)
        except ImportError:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(sys.argv[1])
                text = ""
                for i, page in enumerate(reader.pages):
                    text += f"\\n--- Page {i + 1} ---\\n"
                    text += page.extract_text() or ""
                print(text[:50000] if len(text) > 50000 else text)
            except ImportError:
                print("[PDF_EXTRACTION_FAILED: Install PyMuPDF or PyPDF2]")
            except Exception as e:
                print(f"[PDF_ERROR: {e}]")
        except Exception as e:
            print(f"[PDF_ERROR: {e}]")
        """

        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        task.arguments = ["-c", pythonScript, url.path]

        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe

        do {
            try task.run()
            task.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if let output = String(data: data, encoding: .utf8), !output.isEmpty {
                if output.contains("[PDF_EXTRACTION_FAILED") || output.contains("[PDF_ERROR") {
                    return """
                    [PDF: \(fileName) (\(fileSizeKB)KB)]
                    \(output)
                    Tip: Install PyMuPDF for PDF support: pip3 install pymupdf
                    """
                }

                let lineCount = output.components(separatedBy: "\n").count
                return """
                [PDF: \(fileName) (\(fileSizeKB)KB, ~\(lineCount) lines extracted)]

                \(output)
                """
            }
        } catch {
            debugLog("PDF extraction error: \(error)")
        }

        return """
        [PDF: \(fileName) (\(fileSizeKB)KB)]
        Unable to extract text. Install PyMuPDF: pip3 install pymupdf
        """
    }

    @objc func sendChatMessage(_ sender: Any) {
        let input = chatInputField?.stringValue ?? ""
        // Allow sending if there's text OR attachments
        guard !input.isEmpty || !attachedFiles.isEmpty else { return }
        guard !isChatLoading else { return }

        // Clear input field
        chatInputField?.stringValue = ""

        // Build message content with file attachments
        var messageContent = input
        var displayMessage = input

        if !attachedFiles.isEmpty {
            // Add file contents to the message
            var fileContents: [String] = []
            var fileNames: [String] = []

            for file in attachedFiles {
                fileNames.append(file.name)
                if let content = readFileContents(file.url) {
                    fileContents.append("--- File: \(file.name) ---\n\(content)\n--- End of \(file.name) ---")
                }
            }

            // Format message with attachments
            if !messageContent.isEmpty {
                messageContent += "\n\n"
            }
            messageContent += "Attached files:\n" + fileContents.joined(separator: "\n\n")

            // Display message shows file names
            if !displayMessage.isEmpty {
                displayMessage += " "
            }
            displayMessage += "[Attached: \(fileNames.joined(separator: ", "))]"

            // Clear attachments
            attachedFiles.removeAll()
            updateAttachmentDisplay()
        }

        // Add user message to display
        appendToChatOutput("You: \(displayMessage)\n\n", color: NSColor.systemBlue)

        // Add to message history with full content including files
        chatMessages.append((role: "user", content: messageContent))

        // Show loading indicator with context
        isChatLoading = true
        let isLargeContent = messageContent.count > 10000
        let thinkingMessage = isLargeContent
            ? "AI: Analyzing document (this may take a minute)...\n"
            : "AI: Thinking...\n"
        appendToChatOutput(thinkingMessage, color: NSColor.systemGreen)

        // Send to local AI server
        sendToLocalAI(messages: chatMessages)
    }

    private func sendToLocalAI(messages: [(role: String, content: String)]) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            // Build request
            var messagesArray: [[String: Any]] = []
            for msg in messages {
                messagesArray.append(["role": msg.role, "content": msg.content])
            }

            // Detect if this is a document analysis request (large content)
            let lastMessage = messages.last?.content ?? ""
            let isLargeContent = lastMessage.count > 10000  // More than 10KB of content

            // Use tools model (Hermes-3 8B) for large documents - much faster
            // Use hybrid (72B + Hermes) for complex reasoning tasks
            let model = isLargeContent ? "mageagent:tools" : "mageagent:hybrid"

            let requestBody: [String: Any] = [
                "model": model,
                "messages": messagesArray,
                "max_tokens": 4096,
                "temperature": 0.7
            ]

            guard let jsonData = try? JSONSerialization.data(withJSONObject: requestBody) else {
                self.handleChatError("Failed to encode request")
                return
            }

            // Always use local AI server directly for tool execution
            let urlString = "\(Config.mageagentURL)/v1/chat/completions"
            guard let url = URL(string: urlString) else {
                self.handleChatError("Invalid URL")
                return
            }

            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            // 10 minute timeout for document analysis, 5 min for regular chat
            request.timeoutInterval = isLargeContent ? 600 : 300

            request.httpBody = jsonData

            let task = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
                guard let self = self else { return }

                DispatchQueue.main.async {
                    self.isChatLoading = false

                    // Remove loading indicator line (either "Thinking..." or "Analyzing document...")
                    if let textView = self.chatOutputTextView {
                        var text = textView.string
                        let thinkingMsg = "AI: Thinking...\n"
                        let analyzingMsg = "AI: Analyzing document (this may take a minute)...\n"
                        if text.hasSuffix(thinkingMsg) {
                            text = String(text.dropLast(thinkingMsg.count))
                            textView.string = text
                        } else if text.hasSuffix(analyzingMsg) {
                            text = String(text.dropLast(analyzingMsg.count))
                            textView.string = text
                        }
                    }

                    if let error = error {
                        self.appendToChatOutput("AI Error: \(error.localizedDescription)\n\n", color: NSColor.systemRed)
                        return
                    }

                    guard let data = data else {
                        self.appendToChatOutput("AI: No response received\n\n", color: NSColor.systemRed)
                        return
                    }

                    // Parse response
                    if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        // Check for tool calls that need confirmation
                        if let toolCalls = json["tool_calls"] as? [[String: Any]], !toolCalls.isEmpty {
                            self.handleToolCallsWithConfirmation(toolCalls)
                            return
                        }

                        var responseText: String?

                        // OpenAI format (local AI server uses this)
                        if let choices = json["choices"] as? [[String: Any]],
                           let firstChoice = choices.first,
                           let message = firstChoice["message"] as? [String: Any],
                           let text = message["content"] as? String {
                            responseText = text

                            // Check for tool calls in message
                            if let msgToolCalls = message["tool_calls"] as? [[String: Any]], !msgToolCalls.isEmpty {
                                self.handleToolCallsWithConfirmation(msgToolCalls)
                                return
                            }
                        }
                        // Anthropic format
                        else if let content = json["content"] as? [[String: Any]],
                                let firstContent = content.first,
                                let text = firstContent["text"] as? String {
                            responseText = text
                        }
                        // Error format
                        else if let error = json["error"] as? [String: Any],
                                let message = error["message"] as? String {
                            responseText = "Error: \(message)"
                        }

                        if let text = responseText {
                            // Check if response contains tool call patterns (for models that don't use structured tool calls)
                            let toolPatterns = self.extractToolCallsFromText(text)
                            if !toolPatterns.isEmpty {
                                self.handleToolCallsWithConfirmation(toolPatterns)
                                return
                            }

                            self.appendToChatOutput("AI: \(text)\n\n", color: NSColor.systemGreen)
                            self.chatMessages.append((role: "assistant", content: text))
                        } else {
                            self.appendToChatOutput("AI: Unable to parse response\n\n", color: NSColor.systemRed)
                        }
                    } else {
                        let rawResponse = String(data: data, encoding: .utf8) ?? "Unknown response"
                        self.appendToChatOutput("AI: \(rawResponse)\n\n", color: NSColor.systemOrange)
                    }
                }
            }
            task.resume()
        }
    }

    /// Extract tool calls from text response (for models that output tools as text)
    private func extractToolCallsFromText(_ text: String) -> [[String: Any]] {
        var toolCalls: [[String: Any]] = []

        // Pattern: <tool>ToolName</tool> or ```tool\n{...}\n```
        let patterns = [
            // JSON tool call pattern
            "\\{\\s*\"tool\"\\s*:\\s*\"(\\w+)\"\\s*,\\s*\"arguments\"\\s*:\\s*(\\{[^}]+\\})\\s*\\}",
            // Bash command pattern
            "```bash\\n([^`]+)```",
            "```sh\\n([^`]+)```",
            // Read file pattern
            "<read_file>([^<]+)</read_file>",
            "<Read>([^<]+)</Read>"
        ]

        // Check for bash commands
        if let bashRange = text.range(of: "```bash\n", options: .caseInsensitive) {
            if let endRange = text.range(of: "```", options: [], range: bashRange.upperBound..<text.endIndex) {
                let command = String(text[bashRange.upperBound..<endRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !command.isEmpty {
                    toolCalls.append([
                        "tool": "Bash",
                        "arguments": ["command": command]
                    ])
                }
            }
        }

        // Check for sh commands
        if let shRange = text.range(of: "```sh\n", options: .caseInsensitive) {
            if let endRange = text.range(of: "```", options: [], range: shRange.upperBound..<text.endIndex) {
                let command = String(text[shRange.upperBound..<endRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                if !command.isEmpty && toolCalls.isEmpty {  // Don't duplicate
                    toolCalls.append([
                        "tool": "Bash",
                        "arguments": ["command": command]
                    ])
                }
            }
        }

        return toolCalls
    }

    /// Handle tool calls with user confirmation
    private func handleToolCallsWithConfirmation(_ toolCalls: [[String: Any]]) {
        pendingToolCalls = toolCalls

        for toolCall in toolCalls {
            let toolName = toolCall["tool"] as? String ?? toolCall["function"] as? String ?? "Unknown"
            var argsDescription = ""

            if let args = toolCall["arguments"] as? [String: Any] {
                if let command = args["command"] as? String {
                    argsDescription = command
                } else if let filePath = args["file_path"] as? String {
                    argsDescription = filePath
                } else if let pattern = args["pattern"] as? String {
                    argsDescription = pattern
                } else {
                    argsDescription = args.description
                }
            } else if let argsString = toolCall["arguments"] as? String {
                argsDescription = argsString
            }

            // Show confirmation dialog
            DispatchQueue.main.async { [weak self] in
                self?.showToolConfirmationDialog(tool: toolName, args: argsDescription, toolCall: toolCall)
            }
        }
    }

    /// Show confirmation dialog for tool execution
    private func showToolConfirmationDialog(tool: String, args: String, toolCall: [String: Any]) {
        let alert = NSAlert()
        alert.messageText = "Execute \(tool)?"
        alert.informativeText = "AI wants to execute:\n\n\(tool): \(args)\n\nDo you want to allow this?"
        alert.alertStyle = .warning

        alert.addButton(withTitle: "Execute")
        alert.addButton(withTitle: "Cancel")

        // Add "Always Allow" button for convenience
        // alert.addButton(withTitle: "Always Allow")

        let response = alert.runModal()

        if response == .alertFirstButtonReturn {
            // Execute the tool
            executeToolCall(toolCall)
        } else {
            appendToChatOutput("⚠️ Tool execution cancelled by user\n\n", color: NSColor.systemOrange)
        }
    }

    /// Execute a single tool call
    private func executeToolCall(_ toolCall: [String: Any]) {
        let toolName = toolCall["tool"] as? String ?? toolCall["function"] as? String ?? ""

        guard let args = toolCall["arguments"] as? [String: Any] ?? parseArgsString(toolCall["arguments"] as? String) else {
            appendToChatOutput("❌ Invalid tool arguments\n\n", color: NSColor.systemRed)
            return
        }

        appendToChatOutput("🔧 Executing \(toolName)...\n", color: NSColor.systemCyan)

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }

            var result: String = ""

            switch toolName {
            case "Bash":
                result = self.executeBash(args["command"] as? String ?? "")
            case "Read":
                result = self.executeRead(args["file_path"] as? String ?? "")
            case "Write":
                result = self.executeWrite(
                    filePath: args["file_path"] as? String ?? "",
                    content: args["content"] as? String ?? ""
                )
            case "Glob":
                result = self.executeGlob(
                    pattern: args["pattern"] as? String ?? "*",
                    path: args["path"] as? String ?? NSHomeDirectory()
                )
            case "Grep":
                result = self.executeGrep(
                    pattern: args["pattern"] as? String ?? "",
                    path: args["path"] as? String ?? NSHomeDirectory()
                )
            default:
                result = "Unknown tool: \(toolName)"
            }

            DispatchQueue.main.async {
                // Truncate very long results
                let maxLength = 5000
                var displayResult = result
                if result.count > maxLength {
                    displayResult = String(result.prefix(maxLength)) + "\n... (truncated)"
                }

                self.appendToChatOutput("📋 Result:\n\(displayResult)\n\n", color: NSColor.systemTeal)

                // Add tool result to conversation and continue
                self.chatMessages.append((role: "assistant", content: "Tool result for \(toolName):\n\(result)"))
            }
        }
    }

    private func parseArgsString(_ argsString: String?) -> [String: Any]? {
        guard let str = argsString,
              let data = str.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return json
    }

    // MARK: - Tool Execution Methods

    private func executeBash(_ command: String) -> String {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/bash")
        task.arguments = ["-c", command]

        var environment = ProcessInfo.processInfo.environment
        environment["HOME"] = NSHomeDirectory()
        environment["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
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

            if task.terminationStatus == 0 {
                return output.isEmpty ? "(no output)" : output
            } else {
                return "Exit code: \(task.terminationStatus)\nStdout: \(output)\nStderr: \(errorOutput)"
            }
        } catch {
            return "Error: \(error.localizedDescription)"
        }
    }

    private func executeRead(_ filePath: String) -> String {
        let expandedPath = (filePath as NSString).expandingTildeInPath
        do {
            let content = try String(contentsOfFile: expandedPath, encoding: .utf8)
            return content
        } catch {
            return "Error reading file: \(error.localizedDescription)"
        }
    }

    private func executeWrite(filePath: String, content: String) -> String {
        let expandedPath = (filePath as NSString).expandingTildeInPath
        do {
            try content.write(toFile: expandedPath, atomically: true, encoding: .utf8)
            return "Successfully wrote \(content.count) characters to \(filePath)"
        } catch {
            return "Error writing file: \(error.localizedDescription)"
        }
    }

    private func executeGlob(pattern: String, path: String) -> String {
        let expandedPath = (path as NSString).expandingTildeInPath
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/bash")
        task.arguments = ["-c", "find '\(expandedPath)' -name '\(pattern)' 2>/dev/null | head -100"]

        let outputPipe = Pipe()
        task.standardOutput = outputPipe

        do {
            try task.run()
            task.waitUntilExit()

            let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: outputData, encoding: .utf8) ?? ""
            return output.isEmpty ? "No files found matching '\(pattern)'" : output
        } catch {
            return "Error: \(error.localizedDescription)"
        }
    }

    private func executeGrep(pattern: String, path: String) -> String {
        let expandedPath = (path as NSString).expandingTildeInPath
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/grep")
        task.arguments = ["-r", "-n", "--include=*", pattern, expandedPath]

        let outputPipe = Pipe()
        task.standardOutput = outputPipe

        do {
            try task.run()
            task.waitUntilExit()

            let outputData = outputPipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: outputData, encoding: .utf8) ?? ""
            return output.isEmpty ? "No matches found for '\(pattern)'" : output
        } catch {
            return "Error: \(error.localizedDescription)"
        }
    }

    private func handleChatError(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.isChatLoading = false
            self?.appendToChatOutput("Error: \(message)\n\n", color: NSColor.systemRed)
        }
    }

    private func appendToChatOutput(_ text: String, color: NSColor) {
        guard let textView = chatOutputTextView else { return }

        // Parse and format markdown-style content
        let formattedText = formatMarkdownText(text, baseColor: color)
        textView.textStorage?.append(formattedText)

        // Scroll to bottom
        textView.scrollToEndOfDocument(nil)
    }

    /// Format text with clean, simple markdown rendering
    private func formatMarkdownText(_ text: String, baseColor: NSColor) -> NSAttributedString {
        let result = NSMutableAttributedString()

        // Simple paragraph style
        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.lineSpacing = 3
        paragraphStyle.paragraphSpacing = 4

        // Code block style
        let codeParagraphStyle = NSMutableParagraphStyle()
        codeParagraphStyle.lineSpacing = 2
        codeParagraphStyle.paragraphSpacing = 4
        codeParagraphStyle.headIndent = 8
        codeParagraphStyle.firstLineHeadIndent = 8

        let baseFont = NSFont.systemFont(ofSize: 13)
        let codeFont = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)

        // Simple colors - mostly base color, only code is different
        let codeBackgroundColor = NSColor(calibratedWhite: 0.18, alpha: 1.0)
        let codeTextColor = NSColor(calibratedWhite: 0.85, alpha: 1.0)  // Light gray for code

        let lines = text.components(separatedBy: "\n")
        var inCodeBlock = false
        var codeBlockContent = ""
        var codeBlockLanguage = ""

        for (lineIndex, line) in lines.enumerated() {
            let isLastLine = lineIndex == lines.count - 1

            // Check for code block start/end
            if line.hasPrefix("```") {
                if inCodeBlock {
                    // End code block
                    if !codeBlockContent.isEmpty {
                        // Add small separator before code
                        result.append(NSAttributedString(string: "\n", attributes: [.font: NSFont.systemFont(ofSize: 6)]))

                        // Code content with background
                        let codeAttr = NSAttributedString(string: codeBlockContent + "\n", attributes: [
                            .foregroundColor: codeTextColor,
                            .font: codeFont,
                            .backgroundColor: codeBackgroundColor,
                            .paragraphStyle: codeParagraphStyle
                        ])
                        result.append(codeAttr)
                    }
                    inCodeBlock = false
                    codeBlockContent = ""
                    codeBlockLanguage = ""
                } else {
                    inCodeBlock = true
                    codeBlockLanguage = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                }
                continue
            }

            if inCodeBlock {
                codeBlockContent += (codeBlockContent.isEmpty ? "" : "\n") + line
                continue
            }

            // Regular text - keep it simple, same color
            var processedLine = line

            // Strip markdown headers but keep text
            if line.hasPrefix("### ") {
                processedLine = String(line.dropFirst(4))
            } else if line.hasPrefix("## ") {
                processedLine = String(line.dropFirst(3))
            } else if line.hasPrefix("# ") {
                processedLine = String(line.dropFirst(2))
            }
            // Convert bullet lists to simple bullets
            else if line.hasPrefix("- ") || line.hasPrefix("* ") {
                processedLine = "• " + String(line.dropFirst(2))
            }

            // Strip markdown formatting (backticks, bold, italic)
            processedLine = stripMarkdownFormatting(processedLine)

            let lineAttr = NSAttributedString(string: processedLine, attributes: [
                .foregroundColor: baseColor,
                .font: baseFont,
                .paragraphStyle: paragraphStyle
            ])
            result.append(lineAttr)

            // Add newline
            if !isLastLine {
                result.append(NSAttributedString(string: "\n", attributes: [.font: baseFont]))
            }
        }

        // Handle unclosed code block
        if inCodeBlock && !codeBlockContent.isEmpty {
            let codeAttr = NSAttributedString(string: codeBlockContent + "\n", attributes: [
                .foregroundColor: codeTextColor,
                .font: codeFont,
                .backgroundColor: codeBackgroundColor,
                .paragraphStyle: codeParagraphStyle
            ])
            result.append(codeAttr)
        }

        return result
    }

    /// Strip all markdown formatting markers
    private func stripMarkdownFormatting(_ text: String) -> String {
        var result = text
        // Remove bold markers **text**
        result = result.replacingOccurrences(of: "**", with: "")
        // Remove italic markers *text* (but not bullet points)
        // Only remove * that are surrounded by text
        result = result.replacingOccurrences(of: "\\*([^*]+)\\*", with: "$1", options: .regularExpression)
        // Remove backticks
        result = result.replacingOccurrences(of: "`", with: "")
        // Remove underscores used for emphasis
        result = result.replacingOccurrences(of: "__", with: "")
        return result
    }

    /// Format inline markdown - simplified version (not used, kept for compatibility)
    private func formatInlineMarkdown(_ text: String, baseAttributes: [NSAttributedString.Key: Any], codeFont: NSFont, codeBackground: NSColor) -> NSAttributedString {
        return NSAttributedString(string: text, attributes: baseAttributes)
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
