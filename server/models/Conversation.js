const mongoose = require('mongoose');

const messageSchema = new mongoose.Schema({
  sender: { type: String, enum: ['user', 'bot'], required: true },
  text: { type: String, required: true },
  files: [{ name: { type: String } }],
  timestamp: { type: Date, default: Date.now }
});

const conversationSchema = new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  title: { type: String, default: '' },  // NEW: Conversation title field
  messages: [messageSchema],
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date, default: Date.now }
});

// Update "updatedAt" automatically on save
conversationSchema.pre('save', function (next) {
  this.updatedAt = Date.now();
  next();
});

module.exports = mongoose.model('Conversation', conversationSchema);
