const User = require('../models/User');
const { encrypt } = require('../utils/encryption');

/**
 * GET /api/user/apiKey
 * Retrieves the API key for the authenticated user after decrypting it.
 */
exports.getApiKey = async (req, res) => {
  try {
    const userId = req.user.userId; // Set by the token verification middleware
    const user = await User.findById(userId).select('apiKey');
    if (!user) {
      return res.status(404).json({ success: false, message: 'User not found.' });
    }
    let apiKey;
    if (user.apiKey) {
      apiKey = user.apiKey;
    }
    return res.json({ success: true, apiKey });
  } catch (error) {
    console.error('Error retrieving API key:', error);
    return res.status(500).json({ success: false, message: 'Server error.' });
  }
};

/**
 * POST /api/user/apiKey
 * Updates the API key for the authenticated user. The API key is encrypted before saving.
 */
exports.updateApiKey = async (req, res) => {
  try {
    const userId = req.user.userId;
    const { apiKey } = req.body;
    if (!apiKey || apiKey.trim() === '') {
      return res.status(400).json({ success: false, message: 'API key is required.' });
    }
    // Encrypt the API key before updating the user document.
    const encryptedApiKey = encrypt(apiKey.trim());
    const updatedUser = await User.findByIdAndUpdate(
      userId,
      { apiKey: encryptedApiKey },
      { new: true }
    );
    if (!updatedUser) {
      return res.status(404).json({ success: false, message: 'User not found.' });
    }
    // Optionally, return the decrypted API key in the response.
    return res.json({
      success: true,
      message: 'API key updated successfully.',
      apiKey: decrypt(updatedUser.apiKey),
    });
  } catch (error) {
    console.error('Error updating API key:', error);
    return res.status(500).json({ success: false, message: 'Server error.' });
  }
};
