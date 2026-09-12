const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const mongoose = require('mongoose');
const passport = require('passport');
const LocalStrategy = require('passport-local').Strategy;
const passportLocalMongoose = require('passport-local-mongoose');
const findOrCreate = require('mongoose-findorcreate');
const pathResolver = require('./path_resolver');

const FALLBACK_FILE = pathResolver.resolveRoot('users_offline.json');

// Helper to hash password
function hashPassword(password, salt) {
  return crypto.pbkdf2Sync(password, salt, 1000, 64, 'sha512').toString('hex');
}

// In-Memory / File-backed User Store
class OfflineUserManager {
  constructor() {
    this.users = new Map();
    this.loadUsers();
  }

  loadUsers() {
    try {
      if (fs.existsSync(FALLBACK_FILE)) {
        const raw = fs.readFileSync(FALLBACK_FILE, 'utf8');
        const list = JSON.parse(raw);
        for (const u of list) {
          this.users.set(u.id, u);
        }
      }
    } catch (e) {
      console.log('[DB FALLBACK] Error loading offline users:', e.message);
    }
  }

  saveUsers() {
    try {
      const list = Array.from(this.users.values());
      fs.writeFileSync(FALLBACK_FILE, JSON.stringify(list, null, 2), 'utf8');
    } catch (e) {
      console.log('[DB FALLBACK] Error saving offline users:', e.message);
    }
  }

  register(userObj, password, callback) {
    const username = userObj.username || userObj.email;
    if (!username) return callback(new Error('Username required'));

    // Check if user already exists
    for (const u of this.users.values()) {
      if (u.username.toLowerCase() === username.toLowerCase()) {
        return callback(new Error('A user with the given username is already registered'));
      }
    }

    const salt = crypto.randomBytes(16).toString('hex');
    const hash = hashPassword(password, salt);
    const id = 'offline_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);

    const newUser = {
      id: id,
      _id: id,
      username: username,
      email: username,
      salt: salt,
      hash: hash,
      googleId: userObj.googleId || null
    };

    this.users.set(id, newUser);
    this.saveUsers();
    return callback(null, newUser);
  }

  authenticatePassword(username, password, done) {
    let found = null;
    for (const u of this.users.values()) {
      if (u.username.toLowerCase() === username.toLowerCase()) {
        found = u;
        break;
      }
    }

    if (!found) {
      return done(null, false, { message: 'Incorrect username.' });
    }

    const testHash = hashPassword(password, found.salt);
    if (testHash === found.hash) {
      return done(null, found);
    } else {
      return done(null, false, { message: 'Incorrect password.' });
    }
  }

  findById(id, callback) {
    const u = this.users.get(id);
    if (callback) return callback(null, u || null);
    return u || null;
  }

  findOrCreate(query, callback) {
    for (const u of this.users.values()) {
      if (query.googleId && u.googleId === query.googleId) {
        return callback(null, u);
      }
      if (query.username && u.username === query.username) {
        return callback(null, u);
      }
    }

    const id = 'offline_google_' + Date.now();
    const newUser = {
      id: id,
      _id: id,
      username: query.username || ('google_' + query.googleId),
      googleId: query.googleId,
      salt: '',
      hash: ''
    };
    this.users.set(id, newUser);
    this.saveUsers();
    return callback(null, newUser);
  }
}

async function initializeDatabase(dbUri) {
  let isOffline = false;
  let User = null;

  try {
    console.log('[DB INIT] Attempting connection to MongoDB at:', dbUri || 'mongodb://127.0.0.1:27017/Intrusion');
    
    // Attempt connection with 3-second timeout
    await mongoose.connect(dbUri || 'mongodb://127.0.0.1:27017/Intrusion', {
      useNewUrlParser: true,
      useUnifiedTopology: true,
      serverSelectionTimeoutMS: 3000
    });
    
    console.log('[DB CONNECTED] Successfully connected to MongoDB.');

    const userSchema = new mongoose.Schema({
      email: String,
      password: String,
      googleId: String,
    });

    userSchema.plugin(passportLocalMongoose);
    userSchema.plugin(findOrCreate);

    User = mongoose.models.User || mongoose.model('User', userSchema);
    passport.use(User.createStrategy());
    passport.serializeUser(User.serializeUser());
    passport.deserializeUser(User.deserializeUser());

  } catch (err) {
    console.log('[DB FALLBACK] Local MongoDB not reachable (' + err.message + ').');
    console.log('[DB FALLBACK] Activating Resilient Local Offline Mode.');
    isOffline = true;

    const offlineManager = new OfflineUserManager();

    // Create mock User model adhering to passport interface
    User = {
      register: (userObj, pass, cb) => offlineManager.register(userObj, pass, cb),
      findById: (id, cb) => offlineManager.findById(id, cb),
      findOrCreate: (query, cb) => offlineManager.findOrCreate(query, cb),
      authenticate: () => (username, password, done) => offlineManager.authenticatePassword(username, password, done),
      createStrategy: () => {
        return new LocalStrategy((username, password, done) => {
          offlineManager.authenticatePassword(username, password, done);
        });
      },
      isOfflineMode: true
    };

    passport.use(User.createStrategy());
    passport.serializeUser((user, done) => {
      done(null, user.id || user._id);
    });
    passport.deserializeUser((id, done) => {
      User.findById(id, (err, user) => {
        done(err, user);
      });
    });
  }

  return { User, isOffline };
}

module.exports = { initializeDatabase, OfflineUserManager };
