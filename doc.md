## pdf parser 
4. One important limitation
Your current parser checks the .pdf extension and then asks PyMuPDF to open the file. That's a reasonable starting point, but it is not sufficient security validation for production uploads.

Before exposing this to untrusted users, we'll add explicit file-size limits, content validation, resource limits, and isolated processing. We'll also ensure parser errors are translated into safe, structured application errors at the appropriate boundary.


### tradeoffs 

Component	Responsibility
Upload/API layer	File-size limits and upload restrictions
Format detector	Identify format and reject obvious mismatches
Parser	Validate and extract document contents
Ingestion worker	Enforce processing time and resource limits


#### Step 8: Review the detector's limitations
There are four things worth fixing or explicitly accepting.
1. DOCX validation: Checking for two ZIP entries is a useful structural check, but it doesn't validate the XML or prove the archive is safe. We should eventually add limits on archive size, entry count, and decompressed size to defend against ZIP bombs.
2. TXT validation: Incremental UTF-8 decoding is good. However, reading the entire file without a maximum size still allows an extremely large upload to consume significant processing time.
3. HTML validation: Reading only the first 1 MiB means a valid HTML document with its first recognizable tag later in the file could be rejected. Also, the regular expression is a heuristic, not an HTML parser.
4. PDF validation: Checking %PDF- is an early identification check. PyMuPDF remains responsible for determining whether the document can actually be opened and parsed.