// The renderer is fully bundled, and the Electron main process only uses
// built-in modules. Returning false tells electron-builder not to collect the
// web project's development dependencies into the desktop application.
module.exports = async function beforeBuild() {
  return false;
};
