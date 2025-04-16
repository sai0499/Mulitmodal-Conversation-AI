const express = require('express');
const router = express.Router();
const { getApiKey, updateApiKey } = require('../controllers/userController');
const { verifyToken } = require('../middleware/authMiddleware');

// GET the user's API key
router.get('/apiKey', verifyToken, getApiKey);

// POST a new API key for the user
router.post('/apiKey', verifyToken, updateApiKey);

module.exports = router;
