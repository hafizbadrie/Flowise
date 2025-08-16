import { Embeddings } from '@langchain/core/embeddings'
import { Document } from '@langchain/core/documents'
import { VectorStore } from '@langchain/core/vectorstores'
import { GoogleAuth } from 'google-auth-library'

export interface VertexVectorSearchConfig {
    project: string
    location: string
    indexId: string
    indexEndpointId: string
    auth: GoogleAuth
    textKey?: string
    gcsBucketName: string
}

interface IndexDatapoint {
    datapoint_id: string
    feature_vector: number[]
    restricts?: Array<{ namespace: string; allowList: string[] }>
    crowding_tag?: { crowding_attribute: string }
}

export class VertexVectorSearchStore extends VectorStore {
    declare embeddings: Embeddings
    private config: VertexVectorSearchConfig
    private auth: GoogleAuth
    private textKey: string
    private documentMetadata: Map<string, Document> = new Map()

    _vectorstoreType(): string {
        return 'vertex-vector-search'
    }

    constructor(embeddings: Embeddings, config: VertexVectorSearchConfig) {
        super(embeddings, {})
        this.embeddings = embeddings
        this.config = config
        this.auth = config.auth
        this.textKey = config.textKey || 'text'
    }

    async addDocuments(documents: Document[], options?: { ids?: string[] }): Promise<string[]> {
        const ids = options?.ids || documents.map((_, index) => `doc_${Date.now()}_${index}`)
        const texts = documents.map((doc) => doc.pageContent)
        const embeddings = await this.embeddings.embedDocuments(texts)

        const datapoints: IndexDatapoint[] = documents.map((doc, index) => ({
            datapoint_id: ids[index],
            feature_vector: embeddings[index],
            restricts: doc.metadata?.namespace ? [{ namespace: 'default', allowList: [doc.metadata.namespace] }] : undefined
        }))

        await this.upsertDatapoints(datapoints, documents)
        return ids
    }

    async addVectors(vectors: number[][], documents: Document[], options?: { ids?: string[] }): Promise<string[]> {
        const ids = options?.ids || documents.map((_, index) => `vec_${Date.now()}_${index}`)

        const datapoints: IndexDatapoint[] = documents.map((doc, index) => ({
            datapoint_id: ids[index],
            feature_vector: vectors[index],
            restricts: doc.metadata?.namespace ? [{ namespace: 'default', allowList: [doc.metadata.namespace] }] : undefined
        }))

        await this.upsertDatapoints(datapoints, documents)
        return ids
    }

    async similaritySearchVectorWithScore(query: number[], k: number): Promise<[Document, number][]> {
        console.log(`[VertexVectorSearch] Starting similarity search with k=${k}, query vector length=${query.length}`)
        console.log(`[VertexVectorSearch] Config - Project: ${this.config.project}, Location: ${this.config.location}, IndexEndpointId: ${this.config.indexEndpointId}, IndexId: ${this.config.indexId}`)
        
        const client = await this.auth.getAccessToken()
        console.log(`[VertexVectorSearch] Successfully obtained access token`)
        
        const endpoint = `https://1192351616.asia-southeast1-421203729013.vdb.vertexai.goog/v1/projects/${this.config.project}/locations/${this.config.location}/indexEndpoints/${this.config.indexEndpointId}:findNeighbors`
        console.log(`[VertexVectorSearch] Using endpoint: ${endpoint}`)

        const requestBody = {
            deployed_index_id: this.config.indexId,
            queries: [
                {
                    datapoint: {
                        feature_vector: query
                    },
                    neighbor_count: k
                }
            ]
        }
        console.log(`[VertexVectorSearch] Request body:`, JSON.stringify(requestBody, null, 2))

        console.log(`[VertexVectorSearch] Making API call to Vertex AI Vector Search...`)
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${client}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        })

        console.log(`[VertexVectorSearch] API response status: ${response.status} ${response.statusText}`)
        console.log(`[VertexVectorSearch] Response headers:`, Object.fromEntries(response.headers.entries()))

        if (!response.ok) {
            let errorMessage = `Vertex AI Vector Search API error: ${response.status} ${response.statusText}`
            try {
                const errorBody = await response.text()
                console.log(`[VertexVectorSearch] Error response body:`, errorBody)
                if (errorBody) {
                    errorMessage += `. Server response: ${errorBody}`
                }
            } catch (e) {
                console.warn('[VertexVectorSearch] Failed to read error response body:', e)
            }
            console.error(`[VertexVectorSearch] Final error message: ${errorMessage}`)
            throw new Error(errorMessage)
        }

        const data = await response.json()
        console.log(`[VertexVectorSearch] API response data:`, JSON.stringify(data, null, 2))
        
        const results: [Document, number][] = []

        if (data.nearest_neighbors && data.nearest_neighbors[0]) {
            console.log(`[VertexVectorSearch] Found ${data.nearest_neighbors[0].neighbors?.length || 0} neighbors`)
            for (const neighbor of data.nearest_neighbors[0].neighbors || []) {
                console.log(`[VertexVectorSearch] Processing neighbor with datapoint_id: ${neighbor.datapoint?.datapoint_id}, distance: ${neighbor.distance}`)
                const doc = await this.getDocumentByDatapointId(neighbor.datapoint.datapoint_id)
                if (doc) {
                    const similarityScore = 1 - neighbor.distance
                    console.log(`[VertexVectorSearch] Added document to results with similarity score: ${similarityScore}`)
                    results.push([doc, similarityScore]) // Convert distance to similarity score
                } else {
                    console.warn(`[VertexVectorSearch] Could not retrieve document for datapoint_id: ${neighbor.datapoint?.datapoint_id}`)
                }
            }
        } else {
            console.log(`[VertexVectorSearch] No nearest_neighbors found in response`)
        }

        console.log(`[VertexVectorSearch] Returning ${results.length} results`)
        return results
    }

    async delete(options: { ids: string[] }): Promise<void> {
        const client = await this.auth.getAccessToken()
        const endpoint = `https://${this.config.location}-aiplatform.googleapis.com/v1/projects/${this.config.project}/locations/${this.config.location}/indexes/${this.config.indexId}:removeDatapoints`

        const requestBody = {
            datapoint_ids: options.ids
        }

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${client}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        })

        if (!response.ok) {
            let errorMessage = `Failed to delete datapoints: ${response.status} ${response.statusText}`
            try {
                const errorBody = await response.text()
                if (errorBody) {
                    errorMessage += `. Server response: ${errorBody}`
                }
            } catch (e) {
                console.warn('Failed to read error response body:', e)
            }
            throw new Error(errorMessage)
        }

        // Clean up local metadata store
        options.ids.forEach((id) => {
            this.documentMetadata.delete(id)
        })
    }

    private async upsertDatapoints(datapoints: IndexDatapoint[], documents: Document[]): Promise<void> {
        const client = await this.auth.getAccessToken()
        
        // Store document metadata in our local map for retrieval later
        datapoints.forEach((dp, index) => {
            this.documentMetadata.set(dp.datapoint_id, documents[index])
        })

        // Clean datapoints - only send the required fields to Vertex AI
        const cleanDatapoints = datapoints.map((dp, index) => ({
            datapoint_id: dp.datapoint_id,
            feature_vector: dp.feature_vector,
            // Only include restricts if they contain actual filtering constraints (not metadata)
            ...(dp.restricts && dp.restricts.length > 0 && dp.restricts.some((r) => r.namespace !== 'metadata')
                ? { restricts: dp.restricts.filter((r) => r.namespace !== 'metadata') }
                : {}),
            // Store minimal metadata as crowding tag if needed
            ...(documents[index].metadata?.crowding_attribute
                ? { crowding_tag: { crowding_attribute: documents[index].metadata.crowding_attribute } }
                : {})
        }))

        // Use batch import for all indexes (more reliable than stream updates)
        await this.addDatapointsBatch(cleanDatapoints, client)
    }

    private async addDatapointsBatch(datapoints: any[], client: any): Promise<void> {
        // For Batch index type, first upload data to GCS, then use IndexService.UpdateIndex PATCH API
        // This is the proper way to handle batch imports for Batch index types
        
        // Step 1: Upload datapoints to GCS bucket as JSONL format
        const gcsUri = await this.uploadDatapointsToGCS(datapoints, client)
        
        // Step 2: Use IndexService.UpdateIndex to import from GCS
        const endpoint = `https://${this.config.location}-aiplatform.googleapis.com/v1/projects/${this.config.project}/locations/${this.config.location}/indexes/${this.config.indexId}`
        
        const requestBody = {
            metadata: {
                contentsDeltaUri: gcsUri,
                isCompleteOverwrite: false // Set to true for complete replacement, false for updates
            }
        }

        const response = await fetch(endpoint, {
            method: 'PATCH',
            headers: {
                Authorization: `Bearer ${client}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        })

        if (!response.ok) {
            const errorText = await response.text()
            throw new Error(`Failed to update Batch index via PATCH API: ${response.status} ${response.statusText}. Response: ${errorText}`)
        }

        // The update operation is asynchronous and returns a long-running operation
        const result = await response.json()
        
        if (result.name) {
            console.log(`Batch index update operation started: ${result.name}`)
        } else {
            console.log('Batch index update completed')
        }
    }

    private async uploadDatapointsToGCS(datapoints: any[], client: any): Promise<string> {
        // Convert datapoints to JSONL format required by Vertex AI
        const jsonlData = datapoints.map((dp, index) => {
            // Get the associated document for this datapoint
            const doc = this.documentMetadata.get(dp.datapoint_id)
            
            return JSON.stringify({
                id: dp.datapoint_id,
                embedding: dp.feature_vector,
                // Store document content and metadata in restricts for retrieval
                restricts: [
                    ...(dp.restricts || []),
                    {
                        namespace: 'document_content',
                        allowList: [doc?.pageContent || 'Unknown content']
                    },
                    {
                        namespace: 'document_metadata',
                        allowList: [JSON.stringify(doc?.metadata || {})]
                    }
                ],
                ...(dp.crowding_tag ? { crowding_tag: dp.crowding_tag } : {})
            })
        }).join('\n')

        // Also create a separate metadata file for backup retrieval
        const metadataData = Array.from(this.documentMetadata.entries()).map(([id, doc]) => 
            JSON.stringify({
                datapoint_id: id,
                pageContent: doc.pageContent,
                metadata: doc.metadata
            })
        ).join('\n')

        // Create a directory structure in GCS - Vertex AI expects a directory path
        const timestamp = Date.now()
        const directoryName = `vertex_import_${timestamp}`
        const dataFileName = `data.json`
        const metadataFileName = `metadata.jsonl`
        const dataPath = `${directoryName}/${dataFileName}`
        const metadataPath = `${directoryName}/${metadataFileName}`
        const gcsDirectoryUri = `gs://${this.config.gcsBucketName}/${directoryName}/`
        
        // Upload the main data file
        const dataStorageEndpoint = `https://storage.googleapis.com/upload/storage/v1/b/${this.config.gcsBucketName}/o?uploadType=media&name=${encodeURIComponent(dataPath)}`
        
        const uploadResponse = await fetch(dataStorageEndpoint, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${client}`,
                'Content-Type': 'application/octet-stream'
            },
            body: jsonlData
        })

        if (!uploadResponse.ok) {
            const errorText = await uploadResponse.text()
            throw new Error(`Failed to upload data to GCS: ${uploadResponse.status} ${uploadResponse.statusText}. Response: ${errorText}`)
        }

        // Upload the metadata file for backup retrieval
        const metadataStorageEndpoint = `https://storage.googleapis.com/upload/storage/v1/b/${this.config.gcsBucketName}/o?uploadType=media&name=${encodeURIComponent(metadataPath)}`
        
        const metadataUploadResponse = await fetch(metadataStorageEndpoint, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${client}`,
                'Content-Type': 'application/octet-stream'
            },
            body: metadataData
        })

        if (!metadataUploadResponse.ok) {
            console.warn(`Failed to upload metadata file to GCS: ${metadataUploadResponse.status} ${metadataUploadResponse.statusText}`)
        } else {
            console.log(`Metadata file uploaded to GCS: gs://${this.config.gcsBucketName}/${metadataPath}`)
        }

        console.log(`Data uploaded to GCS directory: ${gcsDirectoryUri}`)
        // Return the directory path, not the file path, as required by Vertex AI Batch API
        return gcsDirectoryUri
    }

    private async getDocumentByDatapointId(datapointId: string): Promise<Document | null> {
        // Return document from our metadata store
        const doc = this.documentMetadata.get(datapointId)
        if (doc) {
            return doc
        }

        // If not found in memory, try to retrieve from GCS metadata backup
        try {
            const metadataFromGCS = await this.retrieveDocumentFromGCSMetadata(datapointId)
            if (metadataFromGCS) {
                // Cache the retrieved document for future use
                this.documentMetadata.set(datapointId, metadataFromGCS)
                return metadataFromGCS
            }
        } catch (error) {
            console.warn(`Failed to retrieve document metadata from GCS for datapoint ${datapointId}:`, error)
        }

        // Fallback: create a minimal document with available information
        return new Document({
            pageContent: `Document retrieved from Vertex AI Vector Search (ID: ${datapointId})`,
            metadata: {
                datapointId: datapointId,
                source: 'vertex-vector-search',
                retrieved: true,
                retrievalMethod: 'fallback'
            }
        })
    }

    private async retrieveDocumentFromGCSMetadata(datapointId: string): Promise<Document | null> {
        try {
            const client = await this.auth.getAccessToken()
            
            // List recent metadata files from GCS bucket
            const listEndpoint = `https://storage.googleapis.com/storage/v1/b/${this.config.gcsBucketName}/o?prefix=vertex_import_&delimiter=/`
            
            const listResponse = await fetch(listEndpoint, {
                headers: {
                    Authorization: `Bearer ${client}`
                }
            })

            if (!listResponse.ok) {
                console.warn(`Failed to list GCS objects: ${listResponse.status}`)
                return null
            }

            const listData = await listResponse.json()
            const prefixes = listData.prefixes || []
            
            // Sort by timestamp (newest first) and check the most recent metadata files
            const sortedPrefixes = prefixes.sort((a: string, b: string) => {
                const timestampA = a.match(/vertex_import_(\d+)/)?.[1] || '0'
                const timestampB = b.match(/vertex_import_(\d+)/)?.[1] || '0'
                return parseInt(timestampB) - parseInt(timestampA)
            }).slice(0, 5) // Check last 5 uploads

            for (const prefix of sortedPrefixes) {
                const metadataPath = `${prefix}metadata.jsonl`
                const downloadEndpoint = `https://storage.googleapis.com/storage/v1/b/${this.config.gcsBucketName}/o/${encodeURIComponent(metadataPath)}?alt=media`
                
                const downloadResponse = await fetch(downloadEndpoint, {
                    headers: {
                        Authorization: `Bearer ${client}`
                    }
                })

                if (downloadResponse.ok) {
                    const metadataContent = await downloadResponse.text()
                    const lines = metadataContent.split('\n').filter(line => line.trim())
                    
                    for (const line of lines) {
                        try {
                            const metadata = JSON.parse(line)
                            if (metadata.datapoint_id === datapointId) {
                                return new Document({
                                    pageContent: metadata.pageContent,
                                    metadata: {
                                        ...metadata.metadata,
                                        datapointId: datapointId,
                                        source: 'vertex-vector-search',
                                        retrieved: true,
                                        retrievalMethod: 'gcs_metadata'
                                    }
                                })
                            }
                        } catch (e) {
                            console.warn('Failed to parse metadata line:', e)
                        }
                    }
                }
            }
            
            return null
        } catch (error) {
            console.warn('Error retrieving document from GCS metadata:', error)
            return null
        }
    }

    static async fromDocuments(
        docs: Document[],
        embeddings: Embeddings,
        config: VertexVectorSearchConfig
    ): Promise<VertexVectorSearchStore> {
        const instance = new VertexVectorSearchStore(embeddings, config)
        await instance.addDocuments(docs)
        return instance
    }

    static async fromExistingIndex(embeddings: Embeddings, config: VertexVectorSearchConfig): Promise<VertexVectorSearchStore> {
        return new VertexVectorSearchStore(embeddings, config)
    }
}
