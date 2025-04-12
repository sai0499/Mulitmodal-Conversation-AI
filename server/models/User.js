const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  matricNumber: {
    type: String,
    unique: true,
    required: true,
  },
  email: {
    type: String,
    lowercase: true,
  },
  password: {
    type: String,
  },
  otpCode: {
    type: String, // We'll store it in plain or hashed form (see below)
  },
  otpExpiry: {
    type: Date,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model('User', userSchema);
