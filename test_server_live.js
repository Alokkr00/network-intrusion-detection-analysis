// Live HTTP verification test for Aegis NIDS server
const http = require('http');
const { spawn } = require('child_process');
const pathResolver = require('./path_resolver');

console.log('[E2E HTTP] Launching app.js test server...');
const serverProcess = spawn('node', ['app.js'], {
  cwd: pathResolver.ROOT_DIR,
  env: { ...process.env, PORT: '3000' },
  stdio: ['ignore', 'pipe', 'pipe']
});

let serverReady = false;

serverProcess.stdout.on('data', (data) => {
  const str = data.toString();
  process.stdout.write('[APP STDOUT] ' + str);
  if (str.includes('Started successfully on port') || str.includes('3000')) {
    if (!serverReady) {
      serverReady = true;
      runHttpTests();
    }
  }
});

serverProcess.stderr.on('data', (data) => {
  process.stderr.write('[APP STDERR] ' + data.toString());
});

serverProcess.on('error', (err) => {
  console.error('[SPAWN ERROR]:', err);
  process.exit(1);
});

// Fallback timeout: check port directly if string match is delayed
const checkInterval = setInterval(() => {
  if (serverReady) return;
  const req = http.get('http://127.0.0.1:3000', (res) => {
    serverReady = true;
    clearInterval(checkInterval);
    runHttpTests();
  });
  req.on('error', () => {});
}, 1000);

const routesToTest = [
  { path: '/', expected: 'Network Intrusion Detection System' },
  { path: '/stats', expected: 'Model Performance & Benchmarks' },
  { path: '/attacks', expected: 'Attack Classifications & Mitigation' },
  { path: '/features', expected: 'Feature Specification' },
  { path: '/about', expected: 'Aegis SOC Architecture' },
  { path: '/login', expected: 'Sign In' },
  { path: '/register', expected: 'Create Account' },
  { path: '/submit', expected: 'Detection Operations' },
  { path: '/parameters', expected: 'Packet Parameter Inspector' },
  { path: '/csv', expected: 'Batch CSV Intrusion Classification' }
];

async function runHttpTests() {
  clearInterval(checkInterval);
  console.log('\n[E2E HTTP] Server is listening. Verifying 10 core routes...');
  let passed = 0;

  for (const route of routesToTest) {
    await new Promise((resolve) => {
      http.get(`http://127.0.0.1:3000${route.path}`, (res) => {
        let body = '';
        res.on('data', (chunk) => body += chunk.toString());
        res.on('end', () => {
          const hasExpected = body.includes(route.expected);
          if (res.statusCode === 200 && hasExpected) {
            console.log(`  [PASS] Route [${route.path}] -> Status: ${res.statusCode}, Content verified ("${route.expected}")`);
            passed++;
          } else {
            console.error(`  [FAIL] Route [${route.path}] -> Status: ${res.statusCode}, Has expected: ${hasExpected}`);
          }
          resolve();
        });
      }).on('error', (e) => {
        console.error(`  [FAIL] Route [${route.path}] -> Error: ${e.message}`);
        resolve();
      });
    });
  }

  console.log(`\n[E2E HTTP] Results: ${passed} / ${routesToTest.length} routes verified.`);
  try {
    serverProcess.kill();
  } catch (e) {}

  if (passed === routesToTest.length) {
    console.log('[E2E HTTP] SUCCESS: All endpoints live and responding cleanly (100% OK).');
    process.exit(0);
  } else {
    console.error('[E2E HTTP] FAILURE: One or more routes failed verification.');
    process.exit(1);
  }
}