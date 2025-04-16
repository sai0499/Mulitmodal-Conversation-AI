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
 * Creates (or updates) a conversation by appending the user's message,
 * generating a bot reply via the Gemini API, and saving both in the database.
 * A conversationId may be passed in to append a message to an existing conversation.
 * Supports file uploads via multer.
 */
exports.sendMessage = async (req, res) => {
  try {
    const userId = req.user.userId;
    const { text, conversationId } = req.body;

    if (!text || text.trim() === '') {
      return res.status(400).json({ success: false, message: 'Message text required' });
    }

    let conversation;
    if (conversationId) {
      conversation = await Conversation.findOne({ _id: conversationId, user: userId });
      if (!conversation) {
        return res.status(404).json({ success: false, message: 'Conversation not found' });
      }
    } else {
      conversation = new Conversation({ user: userId, messages: [] });
      // Generate a conversation title using the Gemini API.
      try {
        const titlePrompt = `Generate a concise and descriptive title for a conversation based on the following user message: "${text}". Give me only the title as reply.`;
        const geminiTitleResponse = await axios.post(
          'http://localhost:5000/api/gemini',
          { text: titlePrompt, skipApiKeyValidation: true },
          { headers: { 'Content-Type': 'application/json' } }
        );
        conversation.title = geminiTitleResponse.data.reply;
      } catch (titleErr) {
        console.error("Error generating conversation title", titleErr);
        conversation.title = text.substring(0, 20) || 'New Chat';
      }
    }

    // Process attached files if any (provided by multer)
    const attachedFiles = req.files
      ? req.files.map(file => ({
          name: file.filename,
          originalName: file.originalname,
          path: file.path,
        }))
      : [];

    // Append the user's message (with file details) to the conversation.
    conversation.messages.push({ sender: 'user', text, files: attachedFiles });

    // Retrieve the user's encrypted API key from the database.
    const userDoc = await User.findById(userId).select('apiKey');
    const encryptedApiKey = (userDoc && userDoc.apiKey) ? userDoc.apiKey : null;

    // Generate the bot reply using the Gemini API.
    // The encrypted API key is sent to the Gemini endpoint, which will decrypt it before using it.
    let botReply = '';
    try {
      const geminiResponse = await axios.post(
        'http://localhost:5000/api/gemini',
        { text },
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
      botReply = `Bot reply to: "${text}"`;
    }

    // Append the bot's reply to the conversation.
    conversation.messages.push({ sender: 'bot', text: botReply, files: [] });

    // Save the conversation.
    await conversation.save();

    return res.json({
      success: true,
      conversationId: conversation._id,
      reply: botReply,
      title: conversation.title,
    });
  } catch (error) {
    console.error('Error sending message:', error);
    return res.status(500).json({ success: false, message: 'Server error' });
  }
};

/**
 * GET /api/conversation/:id
 * Returns a single conversation (including its full message history) for the authenticated user.
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
 * Deletes a conversation (including all its messages) for the authenticated user.
 * Also removes any associated files from disk.
 */
exports.deleteConversation = async (req, res) => {
  try {
    const userId = req.user.userId;
    const { id } = req.params;
    const conversation = await Conversation.findOne({ _id: id, user: userId });
    if (!conversation) {
      return res.status(404).json({ success: false, message: 'Conversation not found' });
    }

    // Remove associated files from disk.
    conversation.messages.forEach(message => {
      if (message.files && message.files.length > 0) {
        message.files.forEach(file => {
          if (file.path) {
            fs.unlink(file.path, err => {
              if (err) {
                console.error(`Error deleting file ${file.path}:`, err);
              } else {
                console.log(`Successfully deleted file: ${file.path}`);
              }
            });
          }
        });
      }
    });

    await Conversation.deleteOne({ _id: id });
    return res.json({
      success: true,
      message: 'Conversation and associated files deleted successfully.',
    });
  } catch (error) {
    console.error("Error deleting conversation:", error);
    return res.status(500).json({ success: false, message: "Server error" });
  }
};
