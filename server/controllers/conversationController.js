const Conversation = require('../models/Conversation');

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
 * Creates (or updates) a conversation by adding the user's message,
 * generating a bot reply, and saving both in the database.
 * Optionally, a conversationId can be passed to append to an existing conversation.
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
    }

    // Append user message
    conversation.messages.push({ sender: 'user', text, files: [] });
    
    // In production, integrate your bot/AI service here.
    // For now, we generate a dummy bot reply.
    const botReply = `Bot reply to: "${text}"`;
    conversation.messages.push({ sender: 'bot', text: botReply, files: [] });

    await conversation.save();

    return res.json({ success: true, conversationId: conversation._id, reply: botReply });
  } catch (error) {
    console.error('Error sending message:', error);
    return res.status(500).json({ success: false, message: 'Server error' });
  }
};

/**
 * GET /api/conversation/:id
 * Returns a conversation (with full message history) for the authenticated user.
 */
exports.getConversationById = async (req, res) => {
    try {
      const userId = req.user.userId;
      const { id } = req.params;
      const conversation = await require('../models/Conversation').findOne({ _id: id, user: userId });
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
 * Deletes a conversation (with all its messages) for the authenticated user.
 */
exports.deleteConversation = async (req, res) => {
    try {
      const userId = req.user.userId;
      const { id } = req.params;
      // Find the conversation and ensure it belongs to the current user
      const conversation = await Conversation.findOne({ _id: id, user: userId });
      if (!conversation) {
        return res.status(404).json({ success: false, message: 'Conversation not found' });
      }
      await Conversation.deleteOne({ _id: id });
      return res.json({ success: true, message: 'Conversation deleted successfully.' });
    } catch (error) {
      console.error("Error deleting conversation:", error);
      return res.status(500).json({ success: false, message: "Server error" });
    }
  };