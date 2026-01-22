void SaveFileCommand::onExecute(Context* context)
{
  Document* document = context->activeDocument();

  // If the document is associated to a file in the file-system, we can
  // save it directly without user interaction.
  if (document->isAssociatedToFile()) {
    const ContextReader reader(context);
    const Document* documentReader = reader.document();

    saveDocumentInBackground(context, documentReader, true);
  }
  // If the document isn't associated to a file, we must to show the
  // save-as dialog to the user to select for first time the file-name
  // for this document.
  else {
    saveAsDialog(context, "Save File");
  }
}
