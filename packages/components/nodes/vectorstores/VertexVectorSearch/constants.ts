export const VERTEX_CONSTANTS = {
    SCOPES: ['https://www.googleapis.com/auth/cloud-platform'],
    DEFAULT_TEXT_KEY: 'text',
    BATCH_PREFIX: 'vertex_import_',
    FILE_NAMES: {
        DATA: 'data.json',
        METADATA: 'metadata.jsonl'
    },
    NAMESPACES: {
        DEFAULT: 'default',
        DOCUMENT_CONTENT: 'document_content',
        DOCUMENT_METADATA: 'document_metadata',
        METADATA: 'metadata'
    },
    ENDPOINTS: {
        AIPLATFORM: (location: string) => `https://${location}-aiplatform.googleapis.com/v1`,
        STORAGE_UPLOAD: (bucketName: string) => `https://storage.googleapis.com/upload/storage/v1/b/${bucketName}/o`,
        STORAGE_LIST: (bucketName: string) => `https://storage.googleapis.com/storage/v1/b/${bucketName}/o`,
        VERTEX_VECTOR_SEARCH: (location: string) => `https://1192351616.asia-southeast1-421203729013.vdb.vertexai.goog/v1`
    },
    ERROR_MESSAGES: {
        MISSING_GCS_BUCKET: 'GCS Bucket Name is required for Batch index operations',
        MISSING_CREDENTIALS: 'Please specify your Google Application Credential',
        UPSERT_ERROR: 'Vertex AI Vector Search upsert error',
        DELETE_ERROR: 'Vertex AI Vector Search delete error'
    },
    DEFAULTS: {
        TOP_K: 4,
        NEIGHBOR_COUNT: 4
    }
}

export const CREDENTIAL_PARAMS = {
    GOOGLE_APP_CREDENTIAL_FILE_PATH: 'googleApplicationCredentialFilePath',
    GOOGLE_APP_CREDENTIAL: 'googleApplicationCredential',
    PROJECT_ID: 'projectID'
}

export const REQUIRED_INPUTS = ['projectId', 'location', 'indexId', 'indexEndpointId', 'gcsBucketName']