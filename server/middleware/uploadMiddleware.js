const multer = require('multer');
const path = require('path');
const fs = require('fs');

// Ensure the uploads directory exists
const uploadDir = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    // Save files in the uploads folder
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    // Generate a unique filename: original base name + timestamp + extension
    const ext = path.extname(file.originalname);
    const basename = path.basename(file.originalname, ext);
    cb(null, `${basename}-${Date.now()}${ext}`);
  },
});

// Define allowed file types – adjust as needed for production use.
const fileFilter = (req, file, cb) => {
  const allowedMimeTypes = [
    'application/pdf',
    'text/plain',
    'image/jpeg',
    'image/png'
  ];
  if (allowedMimeTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Invalid file type. Only PDF, TXT, JPEG, and PNG files are allowed.'));
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
});

module.exports = upload;
