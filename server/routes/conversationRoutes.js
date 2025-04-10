const express = require('express');
const router = express.Router();
const { getConversations, sendMessage, getConversationById, deleteConversation } = require('../controllers/conversationController');
const { verifyToken } = require('../middleware/authMiddleware');

router.get('/conversations', verifyToken, getConversations);
router.get('/conversation/:id', verifyToken, getConversationById);
router.post('/conversation', verifyToken, sendMessage);
router.delete('/conversation/:id', verifyToken, deleteConversation);  // NEW endpoint

module.exports = router;
