// models/User.js
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
    type: String,
  },
  otpExpiry: {
    type: Date,
  },
  apiKey: {
    type: String,
    default: null, // New field for storing the API key
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model('User', userSchema);
