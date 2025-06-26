# Medical Text Analysis API

This application provides a web interface and API for analyzing medical documents (PDFs) to extract medical terms and match them with ICD-10 codes.

## Features

- PDF document upload and text extraction
- Medical term extraction using Medspacy
- ICD-10 code matching with exact and partial matching support
- Interactive web interface for document analysis
- Term validation and status tracking
- Support for various medical entity types (PROBLEM, DIAGNOSIS, TREATMENT, PROCEDURE)

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the required Medspacy model:
```bash
python -m spacy download en_core_sci_lg
```

## Usage

1. Start the FastAPI server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. Use the web interface to:
   - Upload PDF documents
   - View extracted medical terms
   - See matched ICD-10 codes
   - Validate or exclude terms
   - Track term status

## API Endpoints

- `GET /`: Web interface for document analysis
- `POST /analyze`: Analyze PDF document and extract medical terms
  - Request: Multipart form with PDF file
  - Response: JSON with extracted terms and ICD-10 matches

## Data Format

The application uses the following data structures:

### Analysis Results
```json
{
    "filename": "example.pdf",
    "total_terms": 10,
    "results": [
        {
            "term": "hypertension",
            "type": "DIAGNOSIS",
            "icd10_matches": [
                {
                    "code": ["T44.5X1", "T44.5X2"],
                    "match_type": "exact",
                    "substance": "Hypertension medication"
                }
            ],
            "validation_status": "pending"
        }
    ]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
