"""Stealth evasion for Playwright and curl_cffi download tiers.

Provides browser fingerprint spoofing, TLS fingerprint alignment,
and anti-detection hardening for both headless Chromium (Playwright)
and curl_cffi HTTP clients.
"""

# ─── Playwright Stealth ──────────────────────────────────────────

STEALTH_SCRIPTS = """
// 1. navigator.webdriver — return undefined
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
delete navigator.__proto__.webdriver;

// 2. window.chrome — add runtime, loadTimes, csi stubs
window.chrome = {
    runtime: {
        PlatformOs: {MAC: 'mac', WIN: 'win', ANDROID: 'android', CROS: 'cros', LINUX: 'linux', OPENBSD: 'openbsd'},
        PlatformArch: {ARM: 'arm', X86_32: 'x86-32', X86_64: 'x86-64', MIPS: 'mips', MIPS64: 'mips64'},
        PlatformNaclArch: {ARM: 'arm', X86_32: 'x86-32', X86_64: 'x86-64', MIPS: 'mips', MIPS64: 'mips64'},
        RequestUpdateCheckStatus: {THROTTLED: 'throttled', NO_UPDATE: 'no_update', UPDATE_AVAILABLE: 'update_available'},
        OnInstalledReason: {INSTALL: 'install', UPDATE: 'update', CHROME_UPDATE: 'chrome_update', SHARED_MODULE_UPDATE: 'shared_module_update'},
        OnRestartRequiredReason: {APP_UPDATE: 'app_update', OS_UPDATE: 'os_update', PERIODIC: 'periodic'},
    },
    loadTimes: function() {
        return {
            commitLoadTime: Date.now() / 1000 - 1.5,
            connectionInfo: 'h2',
            finishDocumentLoadTime: Date.now() / 1000 - 0.3,
            finishLoadTime: Date.now() / 1000 - 0.1,
            firstPaintAfterLoadTime: 0,
            firstPaintTime: Date.now() / 1000 - 0.9,
            navigationType: 'Other',
            npnNegotiatedProtocol: 'h2',
            requestTime: Date.now() / 1000 - 2.0,
            startLoadTime: Date.now() / 1000 - 1.8,
            wasAlternateProtocolAvailable: false,
            wasFetchedViaSpdy: true,
            wasNpnNegotiated: true,
        };
    },
    csi: function() {
        return {
            onloadT: Date.now(),
            pageT: Date.now() / 1000 - 2.0,
            startE: Date.now(),
            tran: 15,
        };
    },
};

// 3. navigator.plugins — fake PluginArray with Chrome PDF plugins
Object.defineProperty(navigator, 'plugins', {
    get: () => {
        const plugins = [
            {name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format', length: 1},
            {name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '', length: 1},
            {name: 'Native Client', filename: 'internal-nacl-plugin', description: '', length: 2},
        ];
        plugins.item = (i) => plugins[i] || null;
        plugins.namedItem = (name) => plugins.find(p => p.name === name) || null;
        plugins.refresh = () => {};
        return plugins;
    },
});

// 4. navigator.mimeTypes — corresponding MimeTypeArray
Object.defineProperty(navigator, 'mimeTypes', {
    get: () => {
        const mimeTypes = [
            {type: 'application/pdf', suffixes: 'pdf', description: 'Portable Document Format'},
            {type: 'application/x-google-chrome-pdf', suffixes: 'pdf', description: 'Portable Document Format'},
            {type: 'application/x-nacl', suffixes: '', description: 'Native Client Executable'},
            {type: 'application/x-pnacl', suffixes: '', description: 'Portable Native Client Executable'},
        ];
        mimeTypes.item = (i) => mimeTypes[i] || null;
        mimeTypes.namedItem = (name) => mimeTypes.find(m => m.type === name) || null;
        return mimeTypes;
    },
});

// 5. navigator.languages — force en-US, en matching Accept-Language
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});

// 6. navigator.permissions — override query for Notification → 'default'
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) => (
    parameters.name === 'notifications'
        ? Promise.resolve({state: Notification.permission})
        : originalQuery(parameters)
);

// 7. navigator.hardwareConcurrency — return 4 (common laptop)
Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 4});

// 8. navigator.deviceMemory — return 8 (common)
Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});

// 9. navigator.platform — force Win32 matching User-Agent
Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});

// 10. WebGL vendor/renderer — spoof WEBGL_debug_renderer_info
const getParameter = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(parameter) {
    const UNMASKED_VENDOR_WEBGL = 0x9245;
    const UNMASKED_RENDERER_WEBGL = 0x9246;
    if (parameter === UNMASKED_VENDOR_WEBGL) return 'Intel Inc.';
    if (parameter === UNMASKED_RENDERER_WEBGL) return 'Intel Iris OpenGL Engine';
    return getParameter.call(this, parameter);
};

// 11. Notification.permission — return 'default'
Object.defineProperty(Notification, 'permission', {get: () => 'default'});

// 12. iframe contentWindow — patch to not leak automation
const iframeProto = HTMLIFrameElement.prototype;
const origContentWindow = Object.getOwnPropertyDescriptor(iframeProto, 'contentWindow');
if (origContentWindow) {
    Object.defineProperty(iframeProto, 'contentWindow', {
        get: function() {
            const win = origContentWindow.get.call(this);
            if (win) {
                try { Object.defineProperty(win.navigator, 'webdriver', {get: () => undefined}); } catch(e) {}
            }
            return win;
        },
    });
}

// 13. navigator.connection — realistic rtt, downlink, effectiveType
Object.defineProperty(navigator, 'connection', {
    get: () => ({
        rtt: 50,
        downlink: 10,
        effectiveType: '4g',
        saveData: false,
    }),
});
"""

STEALTH_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-infobars",
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
]


def get_stealth_context_options(user_agent: str) -> dict:
    """Return Playwright browser context options with stealth settings."""
    return {
        "viewport": {"width": 1920, "height": 1080},
        "device_scale_factor": 1,
        "has_touch": False,
        "locale": "en-US",
        "timezone_id": "America/Vancouver",
        "user_agent": user_agent,
        "extra_http_headers": {
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
        },
    }


def apply_stealth(page) -> None:
    """Inject stealth scripts into a Playwright page."""
    page.add_init_script(STEALTH_SCRIPTS)


# ─── curl_cffi Stealth ───────────────────────────────────────────

CURL_IMPERSONATE_VERSION = "chrome131"


def get_curl_extra_fingerprints():
    """Return ExtraFingerprints for curl_cffi TLS/HTTP2 hardening."""
    from curl_cffi.requests import ExtraFingerprints

    return ExtraFingerprints(
        tls_grease=True,
        tls_permute_extensions=True,
        http2_stream_weight=256,
        http2_stream_exclusive=1,
    )
