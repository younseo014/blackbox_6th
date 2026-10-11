import { app, BrowserWindow, net, protocol, session, shell } from "electron";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const APP_ORIGIN = "memoryguard://app";

protocol.registerSchemesAsPrivileged([
  {
    scheme: "memoryguard",
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
      stream: true,
    },
  },
]);

function rendererPathFromUrl(requestUrl) {
  const rendererRoot = path.resolve(__dirname, "dist-desktop");
  const url = new URL(requestUrl);
  const pathname = decodeURIComponent(url.pathname === "/" ? "/index.html" : url.pathname);
  const candidate = path.resolve(rendererRoot, `.${pathname}`);

  if (candidate !== rendererRoot && !candidate.startsWith(`${rendererRoot}${path.sep}`)) {
    return null;
  }

  return candidate;
}

async function registerDesktopProtocol() {
  await protocol.handle("memoryguard", (request) => {
    const filePath = rendererPathFromUrl(request.url);

    if (!filePath || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      return new Response("Not found", { status: 404 });
    }

    return net.fetch(pathToFileURL(filePath).toString());
  });
}

function configurePermissions() {
  const isTrustedOrigin = (requestingUrl) =>
    requestingUrl === APP_ORIGIN || requestingUrl.startsWith(`${APP_ORIGIN}/`);
  const isCameraPermission = (permission) => permission === "media";

  session.defaultSession.setPermissionCheckHandler((_webContents, permission, requestingOrigin) => {
    return isCameraPermission(permission) && isTrustedOrigin(requestingOrigin);
  });

  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    callback(isCameraPermission(permission) && isTrustedOrigin(webContents.getURL()));
  });
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 390,
    minHeight: 640,
    backgroundColor: "#f6f8fa",
    title: "메모리 가드",
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  window.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://") || url.startsWith("http://")) {
      void shell.openExternal(url);
    }
    return { action: "deny" };
  });

  void window.loadURL(`${APP_ORIGIN}/index.html`);
}

app.whenReady().then(async () => {
  await registerDesktopProtocol();
  configurePermissions();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
