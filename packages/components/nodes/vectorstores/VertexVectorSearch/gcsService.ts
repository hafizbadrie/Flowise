import { Document } from '@langchain/core/documents'
import { GoogleAuth } from 'google-auth-library'
import { VERTEX_CONSTANTS } from './constants'

interface IndexDatapoint {
    datapoint_id: string
    feature_vector: number[]
    restricts?: Array<{ namespace: string; allowList: string[] }>
    crowding_tag?: { crowding_attribute: string }
}

export class GCSService {
    private bucketName: string
    private auth: GoogleAuth

    constructor(bucketName: string, auth: GoogleAuth) {
        this.bucketName = bucketName
        this.auth = auth
    }

    async uploadDatapoints(datapoints: IndexDatapoint[], documentMetadata: Map<string, Document>): Promise<string> {
        const accessToken = await this.auth.getAccessToken()
        if (!accessToken) {
            throw new Error('Failed to obtain access token')
        }
        
        const jsonlData = this.createDatapointsJsonl(datapoints, documentMetadata)
        const metadataData = this.createMetadataJsonl(documentMetadata)

        const timestamp = Date.now()
        const directoryName = `${VERTEX_CONSTANTS.BATCH_PREFIX}${timestamp}`
        const gcsDirectoryUri = `gs://${this.bucketName}/${directoryName}/`

        await this.uploadToGCS(
            `${directoryName}/${VERTEX_CONSTANTS.FILE_NAMES.DATA}`,
            jsonlData,
            accessToken
        )

        await this.uploadMetadataToGCS(
            `${directoryName}/${VERTEX_CONSTANTS.FILE_NAMES.METADATA}`,
            metadataData,
            accessToken
        )

        console.log(`Data uploaded to GCS directory: ${gcsDirectoryUri}`)
        return gcsDirectoryUri
    }

    async retrieveDocumentMetadata(datapointId: string): Promise<Document | null> {
        try {
            const accessToken = await this.auth.getAccessToken()
            if (!accessToken) {
                throw new Error('Failed to obtain access token')
            }
            const listEndpoint = `${VERTEX_CONSTANTS.ENDPOINTS.STORAGE_LIST(this.bucketName)}?prefix=${VERTEX_CONSTANTS.BATCH_PREFIX}&delimiter=/`
            
            const listResponse = await fetch(listEndpoint, {
                headers: { Authorization: `Bearer ${accessToken}` }
            })

            if (!listResponse.ok) {
                console.warn(`Failed to list GCS objects: ${listResponse.status}`)
                return null
            }

            const listData = await listResponse.json()
            const prefixes = listData.prefixes || []
            
            const sortedPrefixes = this.sortPrefixesByTimestamp(prefixes).slice(0, 5)

            for (const prefix of sortedPrefixes) {
                const document = await this.searchMetadataInPrefix(prefix, datapointId, accessToken)
                if (document) return document
            }
            
            return null
        } catch (error) {
            console.warn('Error retrieving document from GCS metadata:', error)
            return null
        }
    }

    private createDatapointsJsonl(datapoints: IndexDatapoint[], documentMetadata: Map<string, Document>): string {
        return datapoints.map((dp) => {
            const doc = documentMetadata.get(dp.datapoint_id)
            
            return JSON.stringify({
                id: dp.datapoint_id,
                embedding: dp.feature_vector,
                restricts: [
                    ...(dp.restricts || []),
                    {
                        namespace: VERTEX_CONSTANTS.NAMESPACES.DOCUMENT_CONTENT,
                        allowList: [doc?.pageContent || 'Unknown content']
                    },
                    {
                        namespace: VERTEX_CONSTANTS.NAMESPACES.DOCUMENT_METADATA,
                        allowList: [JSON.stringify(doc?.metadata || {})]
                    }
                ],
                ...(dp.crowding_tag ? { crowding_tag: dp.crowding_tag } : {})
            })
        }).join('\n')
    }

    private createMetadataJsonl(documentMetadata: Map<string, Document>): string {
        return Array.from(documentMetadata.entries()).map(([id, doc]) => 
            JSON.stringify({
                datapoint_id: id,
                pageContent: doc.pageContent,
                metadata: doc.metadata
            })
        ).join('\n')
    }

    private async uploadToGCS(path: string, data: string, accessToken: string): Promise<void> {
        const uploadEndpoint = `${VERTEX_CONSTANTS.ENDPOINTS.STORAGE_UPLOAD(this.bucketName)}?uploadType=media&name=${encodeURIComponent(path)}`
        
        const uploadResponse = await fetch(uploadEndpoint, {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${accessToken}`,
                'Content-Type': 'application/octet-stream'
            },
            body: data
        })

        if (!uploadResponse.ok) {
            const errorText = await uploadResponse.text()
            throw new Error(`Failed to upload data to GCS: ${uploadResponse.status} ${uploadResponse.statusText}. Response: ${errorText}`)
        }
    }

    private async uploadMetadataToGCS(path: string, metadataData: string, accessToken: string): Promise<void> {
        try {
            await this.uploadToGCS(path, metadataData, accessToken)
            console.log(`Metadata file uploaded to GCS: gs://${this.bucketName}/${path}`)
        } catch (error) {
            console.warn(`Failed to upload metadata file to GCS: ${error}`)
        }
    }

    private sortPrefixesByTimestamp(prefixes: string[]): string[] {
        return prefixes.sort((a: string, b: string) => {
            const timestampA = a.match(/vertex_import_(\d+)/)?.[1] || '0'
            const timestampB = b.match(/vertex_import_(\d+)/)?.[1] || '0'
            return parseInt(timestampB) - parseInt(timestampA)
        })
    }

    private async searchMetadataInPrefix(prefix: string, datapointId: string, accessToken: string): Promise<Document | null> {
        const metadataPath = `${prefix}${VERTEX_CONSTANTS.FILE_NAMES.METADATA}`
        const downloadEndpoint = `${VERTEX_CONSTANTS.ENDPOINTS.STORAGE_LIST(this.bucketName)}/${encodeURIComponent(metadataPath)}?alt=media`
        
        const downloadResponse = await fetch(downloadEndpoint, {
            headers: { Authorization: `Bearer ${accessToken}` }
        })

        if (!downloadResponse.ok) return null

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
        
        return null
    }
}