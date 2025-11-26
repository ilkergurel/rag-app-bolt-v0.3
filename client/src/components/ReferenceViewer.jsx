import React, { useContext } from 'react';
import { X, FileText, Bookmark } from 'lucide-react';
import { LanguageContext } from '../context/LanguageContext';

/**
When user click the ref, the window showing document
 */

// Utility function to format the content:
// - Preserves paragraph breaks (\n\n)
// - Removes single newlines (\n) within a paragraph, replacing them with a space for smooth flow.
const formatContent = (content) => {
    if (!content) return "";

    // 1. Normalize line endings (handles Windows/Unix differences)
    let normalizedContent = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

    // 2. Split by paragraph breaks (two or more newlines)
    const paragraphs = normalizedContent.split(/\n\n+/); 

    // 3. Within each paragraph, remove single newlines (making lines continuous)
    const cleanParagraphs = paragraphs.map(p => 
        p.trim().replace(/\n/g, ' ')
    );

    // 4. Rejoin them using standardized paragraph breaks (\n\n)
    return cleanParagraphs.filter(p => p.length > 0).join('\n\n');
};


export default function ReferenceViewer({ reference, onClose }) {
  if (!reference) return null;

  const { t } = useContext(LanguageContext);

    const getFilename = (path) => {
    if (!path || typeof path !== 'string') {
      return t('unknownFile');
    }
    // Both Windows (\) and Unix (/) division, divide and get the last item
    const parts = path.split(/[\\/]/);
    return parts.pop() || path; // If pop returns empty, show the original path
  };

  const filename = getFilename(reference.source);
  const formattedContent = formatContent(reference.content); // Use the new formatted content    

    return (
        <div className="w-full lg:w-1/3 max-w-md h-full flex flex-col bg-white dark:bg-slate-900 border-l border-gray-200 dark:border-slate-800 shadow-lg">
            {/* title */}
            <div className="flex-shrink-0 p-4 flex justify-between items-center border-b border-gray-200 dark:border-slate-800">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {t('reference')}
                </h3>
                <button
                    onClick={onClose}
                    className="p-1 rounded-full text-gray-400 dark:text-slate-400 hover:bg-gray-100 dark:hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                    <X size={20} />
                </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-5 space-y-4">
                {/* File Info */}
                <div className="pb-4 border-b border-gray-200 dark:border-slate-700">
                    <div className="flex items-center gap-2 mb-1">
                        <FileText size={16} className="text-gray-500 dark:text-slate-400" />
                        <span className="text-sm font-medium text-gray-500 dark:text-slate-400">{t('file')}</span>
                    </div>
                    <p className="text-sm text-gray-700 dark:text-slate-300 break-all" title={reference.source}>
                        {filename}
                    </p>
                </div>

                {/* Page Info */}
                <div className="pb-4 border-b border-gray-200 dark:border-slate-700">
                    <div className="flex items-center gap-2 mb-1">
                        <Bookmark size={16} className="text-gray-500 dark:text-slate-400" />
                        <span className="text-sm font-medium text-gray-500 dark:text-slate-400">{t('page')}</span>
                    </div>
                    <p className="text-sm text-gray-700 dark:text-slate-300">
                        {reference.page}
                    </p>
                </div>
                
                {/* Text */}
                <div>
                    <div className="flex items-center gap-2 mb-2">
                        <span className="text-sm font-medium text-gray-500 dark:text-slate-400">{t('content')}</span>
                    </div>
                    <p className="text-sm text-gray-800 dark:text-slate-200 bg-gray-50 dark:bg-slate-800 p-3 rounded-lg whitespace-pre-wrap">
                        {formattedContent}
                    </p>
                </div>
            </div>
        </div>
    );
}