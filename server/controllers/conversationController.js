// controllers/conversationController.js

const fs = require('fs');
const axios = require('axios');
const Conversation = require('../models/Conversation');
const User = require('../models/User');

/**
 * GET /api/conversations
 * Returns all conversations for the authenticated user.
 */
exports.getConversations = async (req, res) => {
  try {
    const userId = req.user.userId;
    const conversations = await Conversation.find({ user: userId }).sort({ updatedAt: -1 });
    return res.json({ success: true, conversations });
  } catch (error) {
    console.error('Error fetching conversations:', error);
    return res.status(500).json({ success: false, message: 'Server error' });
  }
};

/**
 * POST /api/conversation
 * Appends the user's message, builds a memory-aware prompt,
 * sends both `text` (the full prompt) and `rawQuery` (just the user text)
 * to the Gemini endpoint, and saves the reply.
 */
exports.sendMessage = async (req, res) => {
  try {
    const userId = req.user.userId;
    const { text, conversationId } = req.body;
    if (!text || text.trim() === '') {
      return res.status(400).json({ success: false, message: 'Message text required' });
    }

    // 1. Load or create the Conversation
    let conversation;
    if (conversationId) {
      conversation = await Conversation.findOne({ _id: conversationId, user: userId });
      if (!conversation) {
        return res.status(404).json({ success: false, message: 'Conversation not found' });
      }
    } else {
      conversation = new Conversation({ user: userId, messages: [] });
      // Generate an initial title
      try {
        const titlePrompt = `Generate a concise and descriptive title for a conversation based on the following user message: "${text}". Give me only the title as reply.`;
        const titleResp = await axios.post(
          process.env.GEMINI_URL || 'http://localhost:5000/api/gemini',
          { text: titlePrompt, skipApiKeyValidation: true },
          { headers: { 'Content-Type': 'application/json' } }
        );
        conversation.title = titleResp.data.reply;
      } catch (titleErr) {
        console.error('Error generating conversation title:', titleErr);
        conversation.title = text.substring(0, 20) || 'New Chat';
      }
    }

    // 2. Handle any uploaded files
    const attachedFiles = req.files
      ? req.files.map(file => ({
          name: file.filename,
          originalName: file.originalname,
          path: file.path,
        }))
      : [];

    // 3. Append the new user message
    conversation.messages.push({ sender: 'user', text, files: attachedFiles });

    // 4. Fetch the user's encrypted API key
    const userDoc = await User.findById(userId).select('apiKey');
    const encryptedApiKey = (userDoc && userDoc.apiKey) ? userDoc.apiKey : null;

    // 5. Build the transcript of the last N turns
    const MAX_TURNS = 20;
    const turns = conversation.messages.slice(-MAX_TURNS);
    const transcript = turns.map(msg => {
      const role = msg.sender === 'user' ? 'User' : 'Assistant';
      return `${role}: ${msg.text}`;
    }).join('\n');
    const fullPrompt = `${transcript}\nAssistant:`;

    // 6. Call Gemini with both fullPrompt and rawQuery
    let botReply = '';
    try {
      const geminiResponse = await axios.post(
        process.env.GEMINI_URL || 'http://localhost:5000/api/gemini',
        {
          text: fullPrompt,
          rawQuery: text
        },
        {
          headers: {
            'Content-Type': 'application/json',
            'X-User-ApiKey': encryptedApiKey
          }
        }
      );
      botReply = geminiResponse.data.reply;
    } catch (err) {
      console.error('Error querying Gemini API:', err);
      botReply = `I’m sorry, I couldn’t process your request right now.`;
    }

    // 7. Append and save
    conversation.messages.push({ sender: 'bot', text: botReply, files: [] });
    await conversation.save();

    // 8. Return the reply
    return res.json({
      success: true,
      conversationId: conversation._id,
      reply: botReply,
      title: conversation.title,
    });
  } catch (error) {
    console.error('Error in sendMessage:', error);
    return res.status(500).json({ success: false, message: 'Server error' });
  }
};

/**
 * GET /api/conversation/:id
 * Returns a single conversation’s full history.
 */
exports.getConversationById = async (req, res) => {
  try {
    const userId = req.user.userId;
    const { id } = req.params;
    const conversation = await Conversation.findOne({ _id: id, user: userId });
    if (!conversation) {
      return res.status(404).json({ success: false, message: 'Conversation not found' });
    }
    return res.json({ success: true, conversation });
  } catch (error) {
    console.error('Error fetching conversation:', error);
    return res.status(500).json({ success: false, message: 'Server error' });
  }
};

/**
 * DELETE /api/conversation/:id
 * Deletes a conversation and any attached files.
 */
exports.deleteConversation = async (req, res) => {
  try {
    const userId = req.user.userId;
    const { id } = req.params;
    const conversation = await Conversation.findOne({ _id: id, user: userId });
    if (!conversation) {
      return res.status(404).json({ success: false, message: 'Conversation not found' });
    }

    conversation.messages.forEach(message => {
      (message.files || []).forEach(file => {
        if (file.path) {
          fs.unlink(file.path, err => {
            if (err) console.error(`Error deleting file ${file.path}:`, err);
          });
        }
      });
    });

    await Conversation.deleteOne({ _id: id });
    return res.json({
      success: true,
      message: 'Conversation and associated files deleted successfully.',
    });
  } catch (error) {
    console.error('Error deleting conversation:', error);
    return res.status(500).json({ success: false, message: 'Server error' });
  }
};
