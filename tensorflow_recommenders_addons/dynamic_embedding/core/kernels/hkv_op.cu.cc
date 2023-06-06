#include "lookup_table_op_registry.h"
#include "merlin_hashtable.cuh"
#include <vector>
#include <cuda.h>
#include <cstdlib>


namespace tensorflow {
namespace recommenders_addons {
namespace hkv {

using namespace lookup_table;

template <class K, class V>
class HKV final : public TFRALookupTableInterface<K, V> {
 private:
    // nv::merlin::HashTable table_;
    std::shared_ptr<nv::merlin::HashTable<K, V> > table_ptr_ = nullptr;
    std::vector<int> key_shape_;
    std::vector<int> value_shape_;
    int64_t dim_;

 public:
    HKV(){}
    HKV(const KVTableInfo& info) {}
    ~HKV(){}

    size_t size() const{
        return table_ptr_->size();
    }
    int64_t dim() const{ 
        return table_ptr_->dim();
    }
    TFRA_Status Find(const K* keys, int64_t key_num, V* values, int64_t value_dim,
                            const V* default_values, int64_t default_value_num) {
        // std::vector<bool> founds(key_num, false);
        std::unique_ptr<bool[]> founds = std::make_unique<bool[]>(key_num);
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        size_t n = key_num;
        table_ptr_->find(n, keys, values, founds.get(), nullptr, stream);
        for (int i=0; i<key_num; i++) {
            if (!founds[i]) {
                CUDA_CHECK(cudaMemcpyAsync((void*)(default_values + i * dim()), (void*)(values + i * dim()), dim() * sizeof(V), cudaMemcpyDeviceToDevice, stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return TFRA_Status::OK();
    }
    TFRA_Status Insert(const K* keys, int64_t key_num, const V* values, int64_t value_dim){
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        table_ptr_->insert_or_assign(key_num, keys, values, nullptr, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return TFRA_Status::OK();
    }
    TFRA_Status Remove(const K* keys, int64_t key_num) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        table_ptr_->erase(key_num, keys, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return TFRA_Status::OK();
    }
    TFRA_Status ImportValues(const K* keys, int64_t key_num,
                                    const V* values, int64_t value_dim) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        table_ptr_->insert_or_assign(key_num, keys, values, nullptr, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return TFRA_Status::OK();
    }
    TFRA_Status ExportValues(K* const keys, V* const values){
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        size_t counter = table_ptr_->export_batch(size(), 0, keys, values, nullptr, stream);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return TFRA_Status::OK();
    }
    TFRA_Status FindWithExists(const K* keys, int64_t key_num,
                                      V* const values, int64_t value_dim,
                                      const V* default_values, int64_t default_value_num, bool* exists) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        size_t n = key_num;
        table_ptr_->find(n, keys, values, exists, nullptr, stream);
        for (int i=0; i<key_num; i++) {
            if (!exists[i]) {
                CUDA_CHECK(cudaMemcpyAsync((void*)(default_values + i * dim()), (void*)(values + i * dim()), dim() * sizeof(V), cudaMemcpyDeviceToDevice, stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return TFRA_Status::OK();
    }

    TFRA_Status Accum(const K* keys, int64_t key_num,
                             const V* values_or_delta, int64_t vod_num, 
                             const bool* exists){
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        std::unique_ptr<bool[]> aoa = std::make_unique<bool[]>(key_num);
        table_ptr_->accum_or_assign(key_num, keys, values_or_delta, aoa.get(), nullptr, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return TFRA_Status::OK();
    }
    TFRA_Status Clear(){
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        table_ptr_->clear(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        return TFRA_Status::OK();
    }
    TFRA_Status Dump(K* const key_buffer, V* const value_buffer, size_t search_offset, size_t buffer_size, size_t* dumped_counter){
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        *dumped_counter = table_ptr_->export_batch(buffer_size, search_offset, key_buffer, value_buffer, nullptr, stream);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return TFRA_Status::OK();
    }
  //   TFRA_Status SaveToFileSystem(const std::string& dirpath, const std::string &file_name, const size_t buffer_size. bool append_to_file){}
  //   TFRA_Status LoadFromFileSystem(const std::string& dirpath, const std::string& file_name. const size_t buffer_size, bool load_entire_dir){}
    TFRA_DataType key_dtype() const{return TFRA_DataTypeToEnum<K>::v();}
    TFRA_DataType value_dtype() const{return TFRA_DataTypeToEnum<V>::v();}
    std::vector<int> key_shape() const{ return key_shape_; }
    std::vector<int> value_shape() const{ return value_shape_; }
    int64_t MemoryUsed() const{ return 0; }
};  // class HKV

#define REGISTER_HKV_KERNEL(key_dtype, value_dtype)                 \
    REGISTER_LOOKUP_TABLE("HKV", "GPU", key_dtype, value_dtype,     \
        HKV<key_dtype, value_dtype>)

REGISTER_HKV_KERNEL(int64_t, float);
// REGISTER_HKV_KERNEL(int64, Eigen::half);
REGISTER_HKV_KERNEL(int64_t, int64_t);
REGISTER_HKV_KERNEL(int64_t, int32);
REGISTER_HKV_KERNEL(int64_t, int8);

REGISTER_HKV_KERNEL(int32, float);

}   // namespace hkv
}   // recommenders_addons
}   // namespace tensorflow