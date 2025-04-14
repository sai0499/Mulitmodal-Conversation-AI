// MarkdownRenderer.jsx
import React from 'react';
import PropTypes from 'prop-types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// Custom renderers with specific CSS class names for styling.
const components = {
  h1: ({ node, children, ...props }) => (
    <h1 className="markdown-heading markdown-h1" {...props}>{children}</h1>
  ),
  h2: ({ node, children, ...props }) => (
    <h2 className="markdown-heading markdown-h2" {...props}>{children}</h2>
  ),
  h3: ({ node, children, ...props }) => (
    <h3 className="markdown-heading markdown-h3" {...props}>{children}</h3>
  ),
  p: ({ node, children, ...props }) => (
    <p className="markdown-paragraph" {...props}>{children}</p>
  ),
  a: ({ node, children, ...props }) => (
    <a
      className="markdown-link"
      {...props}
      target="_blank"
      rel="noopener noreferrer"
    >
      {children}
    </a>
  ),
  ul: ({ node, children, ...props }) => (
    <ul className="markdown-ul" {...props}>{children}</ul>
  ),
  ol: ({ node, children, ...props }) => (
    <ol className="markdown-ol" {...props}>{children}</ol>
  ),
  li: ({ node, children, ...props }) => (
    <li className="markdown-list-item" {...props}>{children}</li>
  ),
};

const MarkdownRenderer = ({ content }) => {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {content}
    </ReactMarkdown>
  );
};

MarkdownRenderer.propTypes = {
  content: PropTypes.string.isRequired,
};

export default MarkdownRenderer;
