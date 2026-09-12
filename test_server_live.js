// Live HTTP verification test for Aegis SOC server
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
  { path: '/', expected: 'Next-Generation Network Intrusion Detection' },
  { path: '/stats', expected: 'Inference Model Performance & Benchmarks' },
  { path: '/attacks', expected: 'Network Intrusion Attack Classifications' },
  { path: '/features', expected: 'Network Traffic Telemetry Features' },
  { path: '/about', expected: 'AEGIS SOC Architecture' },
  { path: '/login', expected: 'SOC Console Sign In' },
  { path: '/register', expected: 'Provision Credentials' },
  { path: '/submit', expected: 'Network Intrusion Detection Operations' },
  { path: '/parameters', expected: 'Custom Packet Parameter Inspector' },
  { path: '/csv', expected: 'Batch CSV Intrusion Classification' }
];

async function runHttpTests() {
  clearInterval(checkInterval);
  console.log('\n[E2E HTTP] Server is listening. Verifying 10 core routes...');

  let passed = 0;
  for (const route of routesToTest) {
    try {
      const result = await fetchRoute(route.path);
      const containsExpected = result.body.toUpperCase().includes(route.expected.toUpperCase());
      if (result.status === 200 && containsExpected) {
        console.log(`  [PASS] Route [${route.path}] -> Status: 200, Content verified ("${route.expected}")`);
        passed++;
      } else {
        console.error(`  [FAIL] Route [${route.path}] -> Status: ${result.status}, Has expected: ${containsExpected}`);
      }
    } catch (err) {
      console.error(`  [FAIL] Route [${route.path}] -> Error: ${err.message}`);
    }
  }

  console.log(`\n[E2E HTTP] Results: ${passed} / ${routesToTest.length} routes verified.`);
  
  // Cleanly shut down
  serverProcess.kill('SIGTERM');
  setTimeout(() => {
    try { serverProcess.kill('SIGKILL'); } catch (e) {}
    process.exit(passed === routesToTest.length ? 0 : 1);
  }, 1000);
}

function fetchRoute(routePath) {
  return new Promise((resolve, reject) => {
    const req = http.get(`http://127.0.0.1:3000${routePath}`, (res) => {
      let body = '';
      res.on('data', chunk => { body += chunk; });
      res.on('end', () => {
        resolve({ status: res.statusCode, body });
      });
    });
    req.on('error', reject);
    req.setTimeout(5000, () => {
      req.destroy(new Error('Request timeout'));
    });
  });
}
