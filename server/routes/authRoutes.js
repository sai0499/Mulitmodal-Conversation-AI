const express = require('express');
const router = express.Router();
const {
  sendOTP,
  verifyOTP,
  signUp,
  login,
  resetPassword,
} = require('../controllers/authController');

// 1. Send OTP (for signup or forgot)
router.post('/send-otp', sendOTP);

// 2. Verify OTP
router.post('/verify-otp', verifyOTP);

// 3. Sign up (finalize after OTP verified)
router.post('/signup', signUp);

// 4. Login
router.post('/login', login);

// 5. Reset Password
router.post('/reset-password', resetPassword);

module.exports = router;
