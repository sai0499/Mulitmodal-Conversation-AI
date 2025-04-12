import React from 'react';
import PropTypes from 'prop-types';
import { MdClose } from 'react-icons/md';
import {
  FaFilePdf,
  FaFileAlt,
  FaFileImage,
  FaFile,
} from 'react-icons/fa';

import './FileAttachment.css';

const FileAttachment = ({ file, removable, onRemove }) => {
  // Log the file to confirm extension logic
  console.log('Rendering FileAttachment for:', file);

  // Safely extract the extension from file.name
  let extension = '';
  if (file.name) {
    extension = file.name.split('.').pop().toLowerCase();
  }

  // Decide which icon or preview to show based on extension
  let filePreview;

  // For images: if we have a previewUrl, display the actual image
  if (['png', 'jpg', 'jpeg', 'gif'].includes(extension)) {
    if (file.previewUrl) {
      filePreview = (
        <img
          src={file.previewUrl}
          alt={file.name}
          className="attachment-thumbnail"
        />
      );
    } else {
      filePreview = <FaFileImage className="attachment-icon" title="Image" />;
    }
  } else if (extension === 'pdf') {
    filePreview = <FaFilePdf className="attachment-icon" title="PDF" />;
  } else if (extension === 'txt') {
    filePreview = <FaFileAlt className="attachment-icon" title="Text File" />;
  } else {
    // Generic file icon
    filePreview = <FaFile className="attachment-icon" title="Generic File" />;
  }

  return (
    <div className="file-attachment">
      <div className="file-preview">
        {filePreview}
      </div>

      <div className="file-details">
        <span className="file-name" title={file.name}>
          {file.name}
        </span>

        {removable && (
          <div className="file-actions">
            <button
              className="remove-btn"
              onClick={onRemove}
              title="Remove file"
            >
              <MdClose />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

FileAttachment.propTypes = {
  file: PropTypes.shape({
    name: PropTypes.string.isRequired,
    type: PropTypes.string,
    previewUrl: PropTypes.string,
  }).isRequired,
  removable: PropTypes.bool,
  onRemove: PropTypes.func,
};

FileAttachment.defaultProps = {
  removable: false,
  onRemove: null,
};

export default FileAttachment;
