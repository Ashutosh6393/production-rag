## pdf parser 
4. One important limitation
Your current parser checks the .pdf extension and then asks PyMuPDF to open the file. That's a reasonable starting point, but it is not sufficient security validation for production uploads.

Before exposing this to untrusted users, we'll add explicit file-size limits, content validation, resource limits, and isolated processing. We'll also ensure parser errors are translated into safe, structured application errors at the appropriate boundary.