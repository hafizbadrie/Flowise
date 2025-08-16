import { flatten } from 'lodash'
import { Embeddings } from '@langchain/core/embeddings'
import { Document } from '@langchain/core/documents'
import { GoogleAuth } from 'google-auth-library'
import { ICommonObject, INode, INodeData, INodeOutputsValue, INodeParams, IndexingResult } from '../../../src/Interface'
import { FLOWISE_CHATID, getBaseClasses, getCredentialData, getCredentialParam } from '../../../src/utils'
import { addMMRInputParams, howToUseFileUpload, resolveVectorStoreOrRetriever } from '../VectorStoreUtils'
import { index } from '../../../src/indexing'
import { VertexVectorSearchStore, VertexVectorSearchConfig } from './core'
import { VERTEX_CONSTANTS, CREDENTIAL_PARAMS, REQUIRED_INPUTS } from './constants'

class VertexVectorSearch_VectorStores implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    badge: string
    baseClasses: string[]
    inputs: INodeParams[]
    credential: INodeParams
    outputs: INodeOutputsValue[]

    constructor() {
        this.label = 'Vertex AI Vector Search'
        this.name = 'vertexVectorSearch'
        this.version = 1.0
        this.type = 'VertexVectorSearch'
        this.icon = 'VertexVectorSearch.svg'
        this.category = 'Vector Stores'
        this.description = `Upsert embedded data and perform similarity search using Google Cloud Vertex AI Vector Search, a fully managed vector database service`
        this.baseClasses = [this.type, 'VectorStoreRetriever', 'BaseRetriever']
        this.credential = {
            label: 'Connect Credential',
            name: 'credential',
            type: 'credential',
            credentialNames: ['googleVertexAuth'],
            optional: true,
            description:
                'Google Vertex AI credential. If you are using a GCP service like Cloud Run, or if you have installed default credentials on your local machine, you do not need to set this credential.'
        }
        this.inputs = [
            {
                label: 'Document',
                name: 'document',
                type: 'Document',
                list: true,
                optional: true
            },
            {
                label: 'Embeddings',
                name: 'embeddings',
                type: 'Embeddings'
            },
            {
                label: 'Record Manager',
                name: 'recordManager',
                type: 'RecordManager',
                description: 'Keep track of the record to prevent duplication',
                optional: true
            },
            {
                label: 'Project ID',
                name: 'projectId',
                type: 'string',
                description: 'Google Cloud Project ID'
            },
            {
                label: 'Location',
                name: 'location',
                type: 'string',
                description: 'Google Cloud location/region (e.g., us-central1)',
                placeholder: 'us-central1'
            },
            {
                label: 'Index ID',
                name: 'indexId',
                type: 'string',
                description: 'Vertex AI Vector Search Index ID'
            },
            {
                label: 'Index Endpoint ID',
                name: 'indexEndpointId',
                type: 'string',
                description: 'Vertex AI Vector Search Index Endpoint ID'
            },
            {
                label: 'GCS Bucket Name',
                name: 'gcsBucketName',
                type: 'string',
                description: 'Google Cloud Storage bucket name for batch import operations (required for Batch index type)',
                placeholder: 'your-gcs-bucket-name'
            },
            {
                label: 'File Upload',
                name: 'fileUpload',
                description: 'Allow file upload on the chat',
                hint: {
                    label: 'How to use',
                    value: howToUseFileUpload
                },
                type: 'boolean',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Text Key',
                name: 'textKey',
                description: 'The key in the metadata for storing text. Default to `text`',
                type: 'string',
                placeholder: 'text',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Metadata Filter',
                name: 'metadataFilter',
                type: 'json',
                optional: true,
                additionalParams: true,
                description: 'Filter to apply to the vector search'
            },
            {
                label: 'Top K',
                name: 'topK',
                description: 'Number of top results to fetch. Default to 4',
                placeholder: '4',
                type: 'number',
                additionalParams: true,
                optional: true
            }
        ]
        addMMRInputParams(this.inputs)
        this.outputs = [
            {
                label: 'Vertex Vector Search Retriever',
                name: 'retriever',
                baseClasses: this.baseClasses
            },
            {
                label: 'Vertex Vector Search Vector Store',
                name: 'vectorStore',
                baseClasses: [this.type, ...getBaseClasses(VertexVectorSearchStore)]
            }
        ]
    }

    private validateRequiredInputs(nodeData: INodeData): void {
        const missing = REQUIRED_INPUTS.filter(field => !nodeData.inputs?.[field])
        if (missing.length) {
            throw new Error(`Missing required fields: ${missing.join(', ')}`)
        }
    }

    private async createAuth(nodeData: INodeData, options: ICommonObject): Promise<GoogleAuth> {
        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const googleApplicationCredentialFilePath = getCredentialParam(CREDENTIAL_PARAMS.GOOGLE_APP_CREDENTIAL_FILE_PATH, credentialData, nodeData)
        const googleApplicationCredential = getCredentialParam(CREDENTIAL_PARAMS.GOOGLE_APP_CREDENTIAL, credentialData, nodeData)
        const projectID = getCredentialParam(CREDENTIAL_PARAMS.PROJECT_ID, credentialData, nodeData)

        const authOptions: ICommonObject = {}
        if (Object.keys(credentialData).length !== 0) {
            if (!googleApplicationCredentialFilePath && !googleApplicationCredential)
                throw new Error(VERTEX_CONSTANTS.ERROR_MESSAGES.MISSING_CREDENTIALS)
            if (googleApplicationCredentialFilePath && !googleApplicationCredential)
                authOptions.keyFile = googleApplicationCredentialFilePath
            else if (!googleApplicationCredentialFilePath && googleApplicationCredential)
                authOptions.credentials = JSON.parse(googleApplicationCredential)

            if (projectID) authOptions.projectId = projectID
        }

        return new GoogleAuth({
            scopes: VERTEX_CONSTANTS.SCOPES,
            ...authOptions
        })
    }

    private async buildConfig(nodeData: INodeData, options: ICommonObject): Promise<VertexVectorSearchConfig> {
        const auth = await this.createAuth(nodeData, options)
        const textKey = nodeData.inputs?.textKey as string
        
        return {
            project: nodeData.inputs?.projectId as string,
            location: nodeData.inputs?.location as string,
            indexId: nodeData.inputs?.indexId as string,
            indexEndpointId: nodeData.inputs?.indexEndpointId as string,
            auth: auth,
            textKey: textKey || VERTEX_CONSTANTS.DEFAULT_TEXT_KEY,
            gcsBucketName: nodeData.inputs?.gcsBucketName as string
        }
    }

    private processDocuments(docs: Document[], isFileUploadEnabled: boolean, chatId?: string): Document[] {
        const flattenDocs = docs && docs.length ? flatten(docs) : []
        const finalDocs = []
        
        for (let i = 0; i < flattenDocs.length; i += 1) {
            if (flattenDocs[i] && flattenDocs[i].pageContent) {
                if (isFileUploadEnabled && chatId) {
                    flattenDocs[i].metadata = { ...flattenDocs[i].metadata, [FLOWISE_CHATID]: chatId }
                }
                finalDocs.push(new Document(flattenDocs[i]))
            }
        }
        
        return finalDocs
    }

    //@ts-ignore
    vectorStoreMethods = {
        async upsert(nodeData: INodeData, options: ICommonObject): Promise<Partial<IndexingResult>> {
            const instance = new VertexVectorSearch_VectorStores()
            instance.validateRequiredInputs(nodeData)
            
            const docs = nodeData.inputs?.document as Document[]
            const embeddings = nodeData.inputs?.embeddings as Embeddings
            const recordManager = nodeData.inputs?.recordManager
            const isFileUploadEnabled = nodeData.inputs?.fileUpload as boolean

            const config = await instance.buildConfig(nodeData, options)
            const finalDocs = instance.processDocuments(docs, isFileUploadEnabled, options.chatId)

            try {
                if (recordManager) {
                    const vectorStore = await VertexVectorSearchStore.fromExistingIndex(embeddings, config)
                    await recordManager.createSchema()
                    const res = await index({
                        docsSource: finalDocs,
                        recordManager,
                        vectorStore,
                        options: {
                            cleanup: recordManager?.cleanup,
                            sourceIdKey: recordManager?.sourceIdKey ?? 'source',
                            vectorStoreName: `vertex_${config.indexId}`
                        }
                    })

                    return res
                } else {
                    await VertexVectorSearchStore.fromDocuments(finalDocs, embeddings, config)
                    return { numAdded: finalDocs.length, addedDocs: finalDocs }
                }
            } catch (e) {
                throw new Error(`${VERTEX_CONSTANTS.ERROR_MESSAGES.UPSERT_ERROR}: ${e}`)
            }
        },

        async delete(nodeData: INodeData, ids: string[], options: ICommonObject): Promise<void> {
            const instance = new VertexVectorSearch_VectorStores()
            instance.validateRequiredInputs(nodeData)
            
            const embeddings = nodeData.inputs?.embeddings as Embeddings
            const recordManager = nodeData.inputs?.recordManager

            const config = await instance.buildConfig(nodeData, options)

            try {
                if (recordManager) {
                    const vectorStoreName = `vertex_${config.indexId}`
                    await recordManager.createSchema()
                    ;(recordManager as any).namespace = (recordManager as any).namespace + '_' + vectorStoreName
                    const keys: string[] = await recordManager.listKeys({})

                    const vertexStore = await VertexVectorSearchStore.fromExistingIndex(embeddings, config)
                    await vertexStore.delete({ ids: keys })
                    await recordManager.deleteKeys(keys)
                } else {
                    const vertexStore = await VertexVectorSearchStore.fromExistingIndex(embeddings, config)
                    await vertexStore.delete({ ids })
                }
            } catch (e) {
                throw new Error(`${VERTEX_CONSTANTS.ERROR_MESSAGES.DELETE_ERROR}: ${e}`)
            }
        }
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        this.validateRequiredInputs(nodeData)
        
        const metadataFilter = nodeData.inputs?.metadataFilter
        const embeddings = nodeData.inputs?.embeddings as Embeddings
        const isFileUploadEnabled = nodeData.inputs?.fileUpload as boolean

        const config = await this.buildConfig(nodeData, options)
        
        console.log(`[VertexVectorSearch] Created config:`, {
            project: config.project,
            location: config.location,
            indexId: config.indexId,
            indexEndpointId: config.indexEndpointId,
            textKey: config.textKey,
            gcsBucketName: config.gcsBucketName
        })

        console.log(`[VertexVectorSearch] Creating vector store from existing index...`)
        const vectorStore = await VertexVectorSearchStore.fromExistingIndex(embeddings, config)

        // Apply filters if specified
        let filter = {}
        if (metadataFilter) {
            filter = typeof metadataFilter === 'object' ? metadataFilter : JSON.parse(metadataFilter)
        }
        if (isFileUploadEnabled && options.chatId) {
            filter = {
                ...filter,
                $or: [
                    ...((filter as any)?.$or || []),
                    { [FLOWISE_CHATID]: { $eq: options.chatId } },
                    { [FLOWISE_CHATID]: { $exists: false } }
                ]
            }
        }

        return resolveVectorStoreOrRetriever(nodeData, vectorStore, filter)
    }
}

module.exports = { nodeClass: VertexVectorSearch_VectorStores }