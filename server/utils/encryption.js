const crypto = require('crypto');

const algorithm = 'aes-256-cbc';
// Derive a 32-byte key from the secret (provide a strong secret in your .env file)
const key = crypto.scryptSync(process.env.ENCRYPTION_SECRET, 'salt', 32);

/**
 * Encrypts a given text.
 * @param {string} text - The plaintext to encrypt.
 * @returns {string} The IV and encrypted text, concatenated with a colon.
 */
function encrypt(text) {
  const iv = crypto.randomBytes(16);
  const cipher = crypto.createCipheriv(algorithm, key, iv);
  let encrypted = cipher.update(text, 'utf8', 'hex');
  encrypted += cipher.final('hex');
  // Return the IV and encrypted data separated by a colon.
  return iv.toString('hex') + ':' + encrypted;
}

module.exports = {
  encrypt,
};
