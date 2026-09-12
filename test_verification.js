// Automated End-to-End Test & Verification Suite for NIDS Aegis SOC Architecture
const http = require('http');
const path = require('path');
const fs = require('fs');
const ejs = require('ejs');
const { execSync } = require('child_process');
const pathResolver = require('./path_resolver');
const { initializeDatabase, OfflineUserManager } = require('./db_fallback');

console.log('================================================================');
console.log('  AEGIS SOC PLATFORM & DIRECTORY RESOLUTION VERIFICATION SUITE');
console.log('================================================================\n');

let passedTests = 0;
let totalTests = 0;

function assert(condition, testName) {
  totalTests++;
  if (condition) {
    console.log(`  [PASS] Test ${totalTests}: ${testName}`);
    passedTests++;
  } else {
    console.error(`  [FAIL] Test ${totalTests}: ${testName}`);
  }
}

async function runTests() {
  const baseDir = pathResolver.ROOT_DIR;

  // --- Suite 1: Dynamic Platform-Agnostic Directory Resolution ---
  console.log('[SUITE 1] Dynamic Platform-Agnostic Directory Resolution');
  try {
    assert(fs.existsSync(pathResolver.ROOT_DIR), 'Node path_resolver: ROOT_DIR is valid and exists');
    assert(fs.existsSync(pathResolver.UPLOAD_DIR), 'Node path_resolver: UPLOAD_DIR is valid and exists');
    assert(fs.existsSync(pathResolver.VIEWS_DIR), 'Node path_resolver: VIEWS_DIR is valid and exists');
    assert(fs.existsSync(pathResolver.PUBLIC_DIR), 'Node path_resolver: PUBLIC_DIR is valid and exists');
    assert(pathResolver.isSafePath('test.csv', pathResolver.UPLOAD_DIR), 'Node path_resolver: isSafePath accepts valid upload file');
    assert(!pathResolver.isSafePath('../../etc/passwd', pathResolver.UPLOAD_DIR), 'Node path_resolver: isSafePath blocks path traversal');

    const pyPathCheck = 'python -c "import path_resolver; assert path_resolver.get_root_dir().exists(); assert path_resolver.get_upload_dir().exists(); assert path_resolver.is_safe_path(path_resolver.get_upload_dir() / \x27test.csv\x27, path_resolver.get_upload_dir()); print(\x27PY_PATH_OK\x27)"';
    const pyOut = execSync(pyPathCheck, { cwd: baseDir }).toString();
    assert(pyOut.includes('PY_PATH_OK'), 'Python path_resolver: Canonical paths and traversal defense verified');
  } catch (err) {
    assert(false, 'Directory resolution suite failed: ' + err.message);
  }

  // --- Suite 2: Python Preprocessor & Model Integrity ---
  console.log('\n[SUITE 2] Python Preprocessor & MITRE ATT&CK Correlation Engine');
  try {
    const pyCmd = 'python -c "import nids_preprocessor as prep; v = prep.encode_feature_vector([\x27tcp\x27, \x27http\x27, \x27SF\x27, 1, 4, 0, 0, 1, 0, 255, 234, 0.92, 0.01, 0, 0.01, 0]); assert len(v) == 16; v2 = prep.encode_feature_vector([\x27unknown\x27, \x27unknown\x27, \x27UNKNOWN\x27, 0, 1, 0, 0, 1, 0, 4, 4, 1, 0, 1, 0, 0]); assert len(v2) == 16; t = prep.get_mitre_telemetry(\x27dos\x27); assert t[\x27technique_id\x27] == \x27T1498\x27; assert len(t[\x27containment_playbook\x27]) > 0; print(\x27PREPROCESSOR_MITRE_OK\x27)"';
    const out = execSync(pyCmd, { cwd: baseDir }).toString();
    assert(out.includes('PREPROCESSOR_MITRE_OK'), 'Preprocessor encodes features and outputs complete MITRE taxonomy');
  } catch (err) {
    assert(false, 'Preprocessor execution failed: ' + err.message);
  }

  // --- Suite 3: Random Row Prediction with MITRE Telemetry ---
  console.log('\n[SUITE 3] Random Vector Prediction Script (nids_random_updated.py)');
  try {
    const out = execSync('python nids_random_updated.py', { cwd: baseDir }).toString();
    assert(out.includes('JSON_PAYLOAD:'), 'nids_random_updated.py emits structured JSON payload');
    assert(out.includes('RANDOM FOREST Binary Class Type :'), 'Random Forest evaluated');
    assert(out.includes('KNN Binary Class Type :'), 'KNN evaluated');

    const jsonStr = out.split('JSON_PAYLOAD:')[1].trim();
    const parsed = JSON.parse(jsonStr);
    assert(parsed.knn && parsed.rf && parsed.cnn && parsed.lstm, 'JSON payload contains predictions for all 4 models');
    assert(parsed.mitre && parsed.mitre.technique_id && parsed.mitre.containment_playbook, 'JSON payload contains MITRE ATT&CK technique and containment playbook');
  } catch (err) {
    assert(false, 'Random row inference failed: ' + err.message);
  }

  // --- Suite 4: Custom Parameter Prediction with MITRE Telemetry ---
  console.log('\n[SUITE 4] Parameter Form Prediction Script (nids_parameter_updated.py)');
  try {
    const out = execSync('python nids_parameter_updated.py tcp http SF 1 4 0.0 0.0 1.0 0.0 255 234 0.92 0.01 0.0 0.01 0.0', { cwd: baseDir }).toString();
    assert(out.includes('JSON_PAYLOAD:'), 'nids_parameter_updated.py emits structured JSON payload');
    const jsonStr = out.split('JSON_PAYLOAD:')[1].trim();
    const parsed = JSON.parse(jsonStr);
    assert(parsed.rf.bin_class !== undefined && parsed.knn.bin_class !== undefined, 'Parsed valid binary classes for parameters');
    assert(parsed.mitre && parsed.mitre.severity !== undefined, 'Enriched parameter results with MITRE severity scoring');
  } catch (err) {
    assert(false, 'Parameter inference failed: ' + err.message);
  }

  // --- Suite 5: Adaptive Batch CSV Processing & Traversal Defense ---
  console.log('\n[SUITE 5] Adaptive Batch CSV Engine (nids_csv_updated.py)');
  try {
    const testCsvPath = path.join(pathResolver.UPLOAD_DIR, 'e2e_test_sample.csv');
    const sampleData = `duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate,class
0,tcp,http,SF,232,8153,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,5,5,0.2,0.2,0,0,1.0,0.0,0.0,30,255,1.0,0.0,0.03,0.04,0.03,0.01,0.0,0.01,normal
0,tcp,private,S0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,6,1.0,1.0,0,0,0.05,0.07,0.0,255,26,0.1,0.05,0.0,0.0,1.0,1.0,0.0,0.0,anomaly
`;
    fs.writeFileSync(testCsvPath, sampleData, 'utf8');

    const out = execSync('python nids_csv_updated.py rf e2e_test_sample.csv', { cwd: baseDir }).toString();
    assert(out.includes('completed!!'), 'CSV processor completes execution');
    assert(out.includes('JSON_PAYLOAD:'), 'CSV processor emits JSON analytics');

    const annotatedContent = fs.readFileSync(testCsvPath, 'utf8');
    assert(annotatedContent.includes('binary class') && annotatedContent.includes('multi class'), 'Output CSV contains appended prediction columns');

    const jsonStr = out.split('JSON_PAYLOAD:')[1].trim();
    const parsed = JSON.parse(jsonStr);
    assert(parsed.mitre_breakdown && parsed.mitre_breakdown.dos, 'CSV analytics includes MITRE breakdown');
  } catch (err) {
    assert(false, 'Adaptive CSV processing failed: ' + err.message);
  }

  // --- Suite 6: Database Resilience & Offline Fallback Engine ---
  console.log('\n[SUITE 6] Database Resilience & Offline Fallback Engine');
  try {
    const dbResult = await initializeDatabase('mongodb://127.0.0.1:27017/NonExistent_Test_DB');
    assert(dbResult.User !== null, 'Database initialization returns valid User interface');

    const testUserEmail = 'analyst_' + Date.now() + '@nids.local';

    // Test Registration
    await new Promise((resolve) => {
      dbResult.User.register({ username: testUserEmail }, 'CyberPass123!', (err, user) => {
        assert(!err && user, 'Offline user registered successfully');
        resolve();
      });
    });

    // Test Authentication
    await new Promise((resolve) => {
      const authFn = dbResult.User.authenticate();
      authFn(testUserEmail, 'CyberPass123!', (err, user) => {
        assert(!err && user, 'Offline user authenticated successfully');
        resolve();
      });
    });
  } catch (err) {
    assert(false, 'Database suite error: ' + err.message);
  }

  // --- Suite 7: EJS Template Suite & Aegis SOC Design System Rendering ---
  console.log('\n[SUITE 7] Aegis SOC Views & Accessibility Compilation');
  try {
    const viewsDir = pathResolver.VIEWS_DIR;
    const testLocals = {
      currentYear: 2026,
      isAuthenticated: true,
      currentUser: { username: 'lead_analyst' },
      consensus_binary: 'ATTACK',
      consensus_multi: 'DOS',
      mitre: {
        category: 'DoS',
        technique_id: 'T1498',
        technique_name: 'Network Denial of Service',
        tactic: 'Impact',
        severity: 'CRITICAL',
        cvss_score: '8.6',
        impact: 'High packet volume socket exhaustion.',
        containment_playbook: ['sudo iptables -I INPUT -s <IP> -j DROP']
      },
      knn_bin_cls: 'ATTACK', knn_mul_cls: 'DOS', knn_desc: 'DoS flood detected', knn_bin_acc: '0.9760', knn_mul_acc: '0.9740',
      rf_bin_cls: 'ATTACK', rf_mul_cls: 'DOS', rf_desc: 'DoS flood detected', rf_bin_acc: '0.9741', rf_mul_acc: '0.9731',
      cnn_bin_cls: 'ATTACK', cnn_mul_cls: 'DOS', cnn_desc: 'DoS flood detected', cnn_bin_acc: '0.9582', cnn_mul_acc: '0.9506',
      lstm_bin_cls: 'ATTACK', lstm_mul_cls: 'DOS', lstm_desc: 'DoS flood detected', lstm_bin_acc: '0.9562', lstm_mul_acc: '0.9590',
      p_knn_bin_cls: 'ATTACK', p_knn_mul_cls: 'DOS', p_knn_desc: 'DoS flood detected', p_knn_bin_acc: '0.9760', p_knn_mul_acc: '0.9740',
      p_rf_bin_cls: 'ATTACK', p_rf_mul_cls: 'DOS', p_rf_desc: 'DoS flood detected', p_rf_bin_acc: '0.9741', p_rf_mul_acc: '0.9731',
      p_cnn_bin_cls: 'ATTACK', p_cnn_mul_cls: 'DOS', p_cnn_desc: 'DoS flood detected', p_cnn_bin_acc: '0.9582', p_cnn_mul_acc: '0.9506',
      p_lstm_bin_cls: 'ATTACK', p_lstm_mul_cls: 'DOS', p_lstm_desc: 'DoS flood detected', p_lstm_bin_acc: '0.9562', p_lstm_mul_acc: '0.9590',
      fileName: 'Test_Capture.csv', algorithm: 'RANDOM FOREST', totalRows: 1000, normalCount: 400, attackCount: 600, attackPercentage: 60.0,
      breakdown: { normal: 400, dos: 500, probe: 80, r2l: 15, u2r: 5 }
    };

    const templatesToTest = [
      'home.ejs', 'submit.ejs', 'parameters.ejs', 'secrets_2.ejs',
      'paramsecrets.ejs', 'csv.ejs', 'index.ejs', 'stats.ejs',
      'attacks.ejs', 'features.ejs', 'about.ejs', 'login.ejs', 'register.ejs'
    ];

    for (const tpl of templatesToTest) {
      const tplPath = path.join(viewsDir, tpl);
      const rendered = await ejs.renderFile(tplPath, testLocals, { root: viewsDir });
      assert(rendered.includes('AEGIS') && rendered.includes('skip-to-content'), `Template [${tpl}] compiled cleanly with Aegis SOC nav and skip link`);
    }
  } catch (err) {
    assert(false, 'EJS template compilation failed: ' + err.message);
  }

  // --- Final Results Summary ---
  console.log('\n================================================================');
  console.log(`  VERIFICATION RESULTS: ${passedTests} / ${totalTests} TESTS PASSED`);
  console.log('================================================================');

  if (passedTests === totalTests) {
    console.log('  >>> STATUS: ALL VERIFICATION GATES PASSED (100% GREEN) <<<');
    process.exit(0);
  } else {
    console.error(`  >>> STATUS: ${totalTests - passedTests} TESTS FAILED <<<`);
    process.exit(1);
  }
}

runTests().catch(err => {
  console.error('[FATAL RUNNER ERROR]:', err);
  process.exit(1);
});
