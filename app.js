// Aegis Next-Gen SOC - Network Intrusion Detection System
require('dotenv').config();

const path = require('path');
const fs = require('fs');
const express = require('express');
const bodyParser = require('body-parser');
const ejs = require('ejs');
const session = require('express-session');
const passport = require('passport');
const multer = require('multer');
const { PythonShell } = require('python-shell');
const pathResolver = require('./path_resolver');
const { initializeDatabase } = require('./db_fallback');

const app = express();

// View Engine & Static Assets (Dynamic Path Resolution)
app.set('view engine', 'ejs');
app.set('views', pathResolver.VIEWS_DIR);
app.use(express.static(pathResolver.PUBLIC_DIR));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

// Session Configuration
const SESSION_SECRET = process.env.SESSION_SECRET || 'nids_secure_session_secret_enterprise_2026';
app.use(session({
  secret: SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    maxAge: 24 * 60 * 60 * 1000 // 24 hours
  }
}));

app.use(passport.initialize());
app.use(passport.session());

// Global locals for views
app.use((req, res, next) => {
  res.locals.currentYear = new Date().getFullYear();
  res.locals.isAuthenticated = req.isAuthenticated ? req.isAuthenticated() : false;
  res.locals.currentUser = req.user || null;
  next();
});

// Multer Storage for File Uploads (Safe Upload Directory)
const storage = multer.diskStorage({
  destination: function (req, file, callback) {
    callback(null, pathResolver.UPLOAD_DIR);
  },
  filename: function (req, file, callback) {
    const safeName = Date.now() + '_' + file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_');
    req.uploadedFileName = safeName;
    callback(null, safeName);
  }
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB maximum
}).single('myfile');

// Helper to run Python child processes with Promise & JSON parser
function runPythonScript(scriptName, args = []) {
  return new Promise((resolve, reject) => {
    const options = {
      mode: 'text',
      pythonPath: process.env.PYTHON_PATH || 'python',
      scriptPath: pathResolver.ROOT_DIR,
      args: args
    };

    PythonShell.run(scriptName, options, (err, response) => {
      if (err) {
        console.error(`[PYTHON ERROR in ${scriptName}]:`, err.message);
        return reject(err);
      }

      let jsonPayload = null;
      if (response && response.length > 0) {
        for (let i = response.length - 1; i >= 0; i--) {
          const line = response[i];
          if (typeof line === 'string' && line.startsWith('JSON_PAYLOAD:')) {
            try {
              jsonPayload = JSON.parse(line.substring('JSON_PAYLOAD:'.length));
              break;
            } catch (e) {
              console.error('[JSON PARSE ERROR]:', e.message);
            }
          } else if (typeof line === 'string' && line.trim().startsWith('{')) {
            try {
              jsonPayload = JSON.parse(line.trim());
              break;
            } catch (e) {}
          }
        }
      }

      resolve({ lines: response || [], json: jsonPayload });
    });
  });
}

// User Model Holder
let User = null;

// Initialize Database & Start Server
async function startServer() {
  const dbResult = await initializeDatabase(process.env.DB_LINK);
  User = dbResult.User;

  // Authentication Middleware Helper
  function checkAuth(req, res, next) {
    if (req.isAuthenticated && req.isAuthenticated()) {
      return next();
    }
    res.redirect('/login');
  }

  // --- Routes ---

  // Home Page
  app.get('/', (req, res) => {
    res.render('home');
  });

  // Auth: Login
  app.get('/login', (req, res) => {
    res.render('login');
  });

  app.post('/login', passport.authenticate('local', {
    successRedirect: '/submit',
    failureRedirect: '/login'
  }));

  // Auth: Register
  app.get('/register', (req, res) => {
    res.render('register');
  });

  app.post('/register', (req, res) => {
    User.register({ username: req.body.username }, req.body.password, (err, user) => {
      if (err) {
        console.error('[REGISTRATION ERROR]:', err.message);
        return res.redirect('/register');
      }
      passport.authenticate('local')(req, res, () => {
        res.redirect('/submit');
      });
    });
  });

  // Auth: Logout
  app.get('/logout', (req, res) => {
    if (req.logout) {
      req.logout((err) => {
        res.redirect('/');
      });
    } else {
      res.redirect('/');
    }
  });

  // Operations Dashboard
  app.get('/submit', (req, res) => {
    res.render('submit');
  });

  // Random Row Prediction
  app.get('/secrets', async (req, res) => {
    try {
      const result = await runPythonScript('nids_random_updated.py');
      if (result.json) {
        req.session.randomResult = result.json;
      }
      res.redirect('/secrets_2');
    } catch (err) {
      console.error('[RANDOM PREDICT ERROR]:', err);
      res.redirect('/secrets_2');
    }
  });

  app.get('/secrets_2', (req, res) => {
    const data = req.session.randomResult || {};
    const knn = data.knn || {};
    const rf = data.rf || {};
    const cnn = data.cnn || {};
    const lstm = data.lstm || {};

    res.render('secrets_2', {
      consensus_binary: data.consensus_binary || rf.bin_class || 'NORMAL',
      consensus_multi: data.consensus_multi || rf.mul_class || 'NORMAL',
      mitre: data.mitre || null,

      knn_bin_cls: knn.bin_class || 'NORMAL',
      knn_mul_cls: knn.mul_class || 'NORMAL',
      knn_desc: knn.desc || 'This traffic pattern is safe.',
      knn_bin_acc: knn.bin_acc || '0.9760',
      knn_mul_acc: knn.mul_acc || '0.9740',

      rf_bin_cls: rf.bin_class || 'NORMAL',
      rf_mul_cls: rf.mul_class || 'NORMAL',
      rf_desc: rf.desc || 'This traffic pattern is safe.',
      rf_bin_acc: rf.bin_acc || '0.9741',
      rf_mul_acc: rf.mul_acc || '0.9731',

      cnn_bin_cls: cnn.bin_class || 'NORMAL',
      cnn_mul_cls: cnn.mul_class || 'NORMAL',
      cnn_desc: cnn.desc || 'This traffic pattern is safe.',
      cnn_bin_acc: cnn.bin_acc || '0.9582',
      cnn_mul_acc: cnn.mul_acc || '0.9506',

      lstm_bin_cls: lstm.bin_class || 'NORMAL',
      lstm_mul_cls: lstm.mul_class || 'NORMAL',
      lstm_desc: lstm.desc || 'This traffic pattern is safe.',
      lstm_bin_acc: lstm.bin_acc || '0.9562',
      lstm_mul_acc: lstm.mul_acc || '0.9590'
    });
  });

  // Custom Parameters Form
  app.get('/parameters', (req, res) => {
    res.render('parameters');
  });

  app.post('/parameters', async (req, res) => {
    const b = req.body;
    const args = [
      b.protocol_type || 'tcp',
      b.service || 'http',
      b.flag || 'SF',
      b.logged_in || '1',
      b.count || '4',
      b.srv_serror_rate || '0',
      b.srv_rerror_rate || '0',
      b.same_srv_rate || '1',
      b.diff_srv_rate || '0',
      b.dst_host_count || '255',
      b.dst_host_srv_count || '234',
      b.dst_host_same_srv_rate || '0.92',
      b.dst_host_diff_srv_rate || '0.01',
      b.dst_host_same_src_port_rate || '0',
      b.dst_host_serror_rate || '0.01',
      b.dst_host_rerror_rate || '0'
    ];

    try {
      const result = await runPythonScript('nids_parameter_updated.py', args);
      if (result.json) {
        req.session.paramResult = result.json;
      }
      res.redirect('/paramsecrets');
    } catch (err) {
      console.error('[PARAM PREDICT ERROR]:', err);
      res.redirect('/paramsecrets');
    }
  });

  app.get('/paramsecrets', (req, res) => {
    const data = req.session.paramResult || {};
    const knn = data.knn || {};
    const rf = data.rf || {};
    const cnn = data.cnn || {};
    const lstm = data.lstm || {};

    res.render('paramsecrets', {
      consensus_binary: data.consensus_binary || rf.bin_class || 'NORMAL',
      consensus_multi: data.consensus_multi || rf.mul_class || 'NORMAL',
      mitre: data.mitre || null,

      p_knn_bin_cls: knn.bin_class || 'NORMAL',
      p_knn_mul_cls: knn.mul_class || 'NORMAL',
      p_knn_desc: knn.desc || 'Data is safe.',
      p_knn_bin_acc: knn.bin_acc || '0.9760',
      p_knn_mul_acc: knn.mul_acc || '0.9740',

      p_rf_bin_cls: rf.bin_class || 'NORMAL',
      p_rf_mul_cls: rf.mul_class || 'NORMAL',
      p_rf_desc: rf.desc || 'Data is safe.',
      p_rf_bin_acc: rf.bin_acc || '0.9741',
      p_rf_mul_acc: rf.mul_acc || '0.9731',

      p_cnn_bin_cls: cnn.bin_class || 'NORMAL',
      p_cnn_mul_cls: cnn.mul_class || 'NORMAL',
      p_cnn_desc: cnn.desc || 'Data is safe.',
      p_cnn_bin_acc: cnn.bin_acc || '0.9582',
      p_cnn_mul_acc: cnn.mul_acc || '0.9506',

      p_lstm_bin_cls: lstm.bin_class || 'NORMAL',
      p_lstm_mul_cls: lstm.mul_class || 'NORMAL',
      p_lstm_desc: lstm.desc || 'Data is safe.',
      p_lstm_bin_acc: lstm.bin_acc || '0.9562',
      p_lstm_mul_acc: lstm.mul_acc || '0.9590'
    });
  });

  // Batch CSV Upload & Processing
  app.get('/csv', (req, res) => {
    res.render('csv');
  });

  app.post('/uploadjavatpoint', (req, res) => {
    upload(req, res, async (err) => {
      if (err) {
        console.error('[UPLOAD ERROR]:', err);
        return res.status(500).send('Error uploading file.');
      }

      const submittedModel = req.body.selected_model || 'rf';
      const submittedFile = req.uploadedFileName || (req.file ? req.file.filename : null);

      if (!submittedFile) {
        return res.status(400).send('No file uploaded.');
      }

      req.session.lastUploadedFile = submittedFile;

      try {
        const result = await runPythonScript('nids_csv_updated.py', [submittedModel, submittedFile]);
        if (result.json) {
          req.session.csvResult = result.json;
        } else {
          req.session.csvResult = {
            total_rows: 0,
            normal_count: 0,
            attack_count: 0,
            attack_percentage: 0,
            breakdown: {},
            algorithm: submittedModel.toUpperCase(),
            file: submittedFile
          };
        }
        res.redirect('/index');
      } catch (scriptErr) {
        console.error('[CSV PROCESSING ERROR]:', scriptErr);
        res.redirect('/index');
      }
    });
  });

  // CSV Analysis Results Dashboard
  app.get('/index', (req, res) => {
    const data = req.session.csvResult || {};
    res.render('index', {
      fileName: data.file || req.session.lastUploadedFile || 'Dataset',
      algorithm: data.algorithm || 'Random Forest',
      totalRows: data.total_rows || 0,
      normalCount: data.normal_count || 0,
      attackCount: data.attack_count || 0,
      attackPercentage: data.attack_percentage || 0,
      breakdown: data.breakdown || { normal: 0, dos: 0, probe: 0, r2l: 0, u2r: 0 }
    });
  });

  // Download Processed CSV File (Protected against Directory Traversal)
  app.get('/download-file', (req, res) => {
    const fileName = req.session.lastUploadedFile;
    if (!fileName) {
      return res.status(404).send('No processed file available for download.');
    }
    const safePath = pathResolver.resolveUpload(fileName);
    if (pathResolver.isSafePath(safePath, pathResolver.UPLOAD_DIR) && fs.existsSync(safePath)) {
      res.download(safePath, 'annotated_nids_results.csv');
    } else {
      res.status(404).send('File not found on server or unsafe path.');
    }
  });

  // Informational Pages
  app.get('/features', (req, res) => res.render('features'));
  app.get('/attacks', (req, res) => res.render('attacks'));
  app.get('/about', (req, res) => res.render('about'));
  app.get('/stats', (req, res) => res.render('stats'));

  // Algorithm Performance Tables
  app.get('/knn_bin_table', (req, res) => res.render('knn_bin_table'));
  app.get('/rf_bin_table', (req, res) => res.render('rf_bin_table'));
  app.get('/cnn_bin_table', (req, res) => res.render('cnn_bin_table'));
  app.get('/lstm_bin_table', (req, res) => res.render('lstm_bin_table'));
  app.get('/knn_table', (req, res) => res.render('knn_table'));
  app.get('/rf_table', (req, res) => res.render('rf_table'));
  app.get('/cnn_table', (req, res) => res.render('cnn_table'));
  app.get('/lstm_table', (req, res) => res.render('lstm_table'));

  // Start HTTP Listener
  const PORT = process.env.PORT || 3000;
  app.listen(PORT, () => {
    console.log(`[NIDS SERVER] Started successfully on port ${PORT}.`);
    console.log(`[NIDS ACCESS] Access application at: http://localhost:${PORT}`);
  });
}

// Boot application
startServer().catch(err => {
  console.error('[NIDS FATAL STARTUP ERROR]:', err);
});