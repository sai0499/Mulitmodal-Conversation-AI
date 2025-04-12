const express = require('express');
const router = express.Router();
const { getConversations, sendMessage, getConversationById, deleteConversation } = require('../controllers/conversationController');
const { verifyToken } = require('../middleware/authMiddleware');
const upload = require('../middleware/uploadMiddleware');

// GET all conversations for a user.
router.get('/conversations', verifyToken, getConversations);

// GET a specific conversation.
router.get('/conversation/:id', verifyToken, getConversationById);

// POST a new message (or update an existing conversation) – now with file upload support.
router.post('/conversation', verifyToken, upload.array('files'), sendMessage);

// DELETE a conversation.
router.delete('/conversation/:id', verifyToken, deleteConversation);

module.exports = router;