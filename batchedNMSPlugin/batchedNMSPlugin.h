/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_BATCHED_NMS_PLUGIN_H
#define TRT_BATCHED_NMS_PLUGIN_H
#include "batchedNMSInference.h"
#include "gatherNMSOutputs.h"
#include "kernel.h"
#include "nmsUtils.h"
#include "plugin.h"
#include <string>
#include <vector>

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

class BatchedNMSPlugin final : public IPluginV2DynamicExt
{
public:
    BatchedNMSPlugin(NMSParameters param);

    BatchedNMSPlugin(const void* data, size_t length);

    ~BatchedNMSPlugin() override = default;

    int getNbOutputs() const override;

    // Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
// DynamicExt plugins returns DimsExprs class instead of Dims
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override;


    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, 
              const void* const* inputs, void* const* outputs, 
              void* workspace, 
              cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    // void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    //     const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    //     const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    // bool supportsFormat(DataType type, PluginFormat format) const override;
      // DynamicExt plugin supportsFormat update.
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2DynamicExt* clone() const override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const override;

    void setPluginNamespace(const char* libNamespace) override;

    const char* getPluginNamespace() const override;


    void setClipParam(bool clip);

    void configurePlugin(const DynamicPluginTensorDesc* inputDims, int nbInputs, 
                       const DynamicPluginTensorDesc* out, int nbOutputs) override;



private:
    NMSParameters param{};
    int boxesSize{};
    int scoresSize{};
    int batchSize{};
    int numPriors{};
    std::string mNamespace;
    bool mClipBoxes{};
    const char* mPluginNamespace;
};

class BatchedNMSPluginCreator : public BaseCreator
{
public:
    BatchedNMSPluginCreator();

    ~BatchedNMSPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    NMSParameters params;
    static std::vector<PluginField> mPluginAttributes;
    bool mClipBoxes;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_BATCHED_NMS_PLUGIN_H
