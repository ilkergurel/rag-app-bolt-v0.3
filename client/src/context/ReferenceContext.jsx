import { createContext } from 'react';
/**
This context triggers the opening of the side window (ReferenceViewer) (Chat) when a reference link (ChatArea) is clicked.
 */

export const ReferenceContext = createContext({
  reference: null,
  setReference: () => {},
});