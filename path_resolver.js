/**
 * Dynamic Platform-Agnostic Directory Resolution Module (Node.js)
 * Aegis SOC Architecture - Network Intrusion Detection System
 * 
 * Provides centralized, normalized, cross-platform directory and path resolution.
 * Supports environment variable overrides and directory traversal security validation.
 */

const path = require('path');
const fs = require('fs');

// Determine Application Root Directory
const ROOT_DIR = process.env.NIDS_ROOT_DIR
  ? path.resolve(process.env.NIDS_ROOT_DIR)
  : path.resolve(__dirname);

// Determine Configured Subdirectories with Fallbacks
const UPLOAD_DIR = process.env.NIDS_UPLOAD_DIR
  ? path.resolve(process.env.NIDS_UPLOAD_DIR)
  : path.resolve(ROOT_DIR, 'Uploaded_files');

const MODELS_DIR = process.env.NIDS_MODELS_DIR
  ? path.resolve(process.env.NIDS_MODELS_DIR)
  : ROOT_DIR;

const DATA_DIR = process.env.NIDS_DATA_DIR
  ? path.resolve(process.env.NIDS_DATA_DIR)
  : ROOT_DIR;

const VIEWS_DIR = process.env.NIDS_VIEWS_DIR
  ? path.resolve(process.env.NIDS_VIEWS_DIR)
  : path.resolve(ROOT_DIR, 'views');

const PUBLIC_DIR = process.env.NIDS_PUBLIC_DIR
  ? path.resolve(process.env.NIDS_PUBLIC_DIR)
  : path.resolve(ROOT_DIR, 'public');

/**
 * Ensure a directory exists synchronously, creating parent directories if needed.
 * @param {string} dirPath 
 * @returns {string} The normalized directory path
 */
function ensureDirExists(dirPath) {
  const resolved = path.resolve(dirPath);
  if (!fs.existsSync(resolved)) {
    fs.mkdirSync(resolved, { recursive: true });
  }
  return resolved;
}

// Ensure UPLOAD_DIR exists on initialization
ensureDirExists(UPLOAD_DIR);

/**
 * Verify whether a target path is safely contained within a base directory
 * to prevent directory traversal attacks (e.g. '../' escapes).
 * @param {string} targetPath 
 * @param {string} baseDir 
 * @returns {boolean}
 */
function isSafePath(targetPath, baseDir = UPLOAD_DIR) {
  const resolvedTarget = path.resolve(baseDir, targetPath);
  const resolvedBase = path.resolve(baseDir);
  return resolvedTarget.startsWith(resolvedBase);
}

/**
 * Convert Windows backslashes to POSIX forward slashes for URLs or cross-platform strings.
 * @param {string} filePath 
 * @returns {string}
 */
function toPosix(filePath) {
  return filePath.replace(/\\/g, '/');
}

/**
 * Resolve a path relative to the application root directory.
 * @param  {...string} segments 
 * @returns {string}
 */
function resolveRoot(...segments) {
  return path.resolve(ROOT_DIR, ...segments);
}

/**
 * Resolve a file within the upload directory safely.
 * @param {string} fileName 
 * @returns {string|null} Resolved path or null if unsafe
 */
function resolveUpload(fileName) {
  const safeName = path.basename(fileName);
  return path.resolve(UPLOAD_DIR, safeName);
}

module.exports = {
  ROOT_DIR,
  UPLOAD_DIR,
  MODELS_DIR,
  DATA_DIR,
  VIEWS_DIR,
  PUBLIC_DIR,
  ensureDirExists,
  isSafePath,
  toPosix,
  resolveRoot,
  resolveUpload
};
