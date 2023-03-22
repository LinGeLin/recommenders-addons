#include "tensorflow_recommenders_addons/dynamic_embedding/core/kernels/lookup_table_op_registry.h"
#include "tensorflow/core/framework/attr_value_util.h"

namespace tensorflow {
namespace recommenders_addons {
namespace op_registry {


LookupTableInterface* op_registry::LookupTableOpRegistry::PtrOpKernelTableFactory::Create(
  OpKernelContext *ctx, OpKernel *kernel) {
    return (*create_func_)(ctx, kernel);
}

template<typename K, typename V>
void op_registry::LookupTableOpRegistry::DeferRegister(const std::string& op_name,
                                                       const std::string& device,
                                                       LookupTableInterface* (create_fn)(OpKernelContext *, OpKernel *)) {
  
  auto factory = std::make_unique<PtrOpKernelTableFactory>(create_fn);
  std::string key_string = tensorflow::DataTypeString(DataTypeToEnum<K>::v());
  std::string value_string = tensorflow::DataTypeString(DataTypeToEnum<V>::v());
  std::string key = op_name + "_" + key_string + "_" + value_string;
  
  DeferRegistrationData_.insert(std::make_pair(key, std::move(factory)));
}

LookupTableInterface *op_registry::LookupTableOpRegistry::LookUp(const string &lookup_table_name,
                                                                 OpKernelContext *ctx, OpKernel *kernel) {
  auto table_find = DeferRegistrationData_.find(lookup_table_name);
  if (table_find == DeferRegistrationData_.end()) {
    return nullptr;
  } else {
    return table_find->second->Create(ctx, kernel);
  }
}

// template<typename K, typename V>
// op_registry::LookupTableOpRegistry* op_registry::LookupTableOpRegistry::Global() {
//   static op_registry::LookupTableOpRegistry tfra_global_op_registry;
//   return &tfra_global_op_registry;
// }

#define PRE_DECLARATION(key_dtype, value_dtype)                                                \
  template void op_registry::LookupTableOpRegistry::DeferRegister<key_dtype, value_dtype>(     \
      const std::string &op_name, const std::string& device,                                   \
      LookupTableInterface* (create_fn)(OpKernelContext *, OpKernel *));

PRE_DECLARATION(int32, double);
PRE_DECLARATION(int32, float);
PRE_DECLARATION(int32, int32);
PRE_DECLARATION(int64_t, double);
PRE_DECLARATION(int64_t, float);
PRE_DECLARATION(int64_t, int32);
PRE_DECLARATION(int64_t, int64_t);
PRE_DECLARATION(int64_t, tstring);
PRE_DECLARATION(int64_t, int8);
PRE_DECLARATION(int64_t, Eigen::half);
PRE_DECLARATION(int64_t, bfloat16)
PRE_DECLARATION(tstring, bool);
PRE_DECLARATION(tstring, double);
PRE_DECLARATION(tstring, float);
PRE_DECLARATION(tstring, int32);
PRE_DECLARATION(tstring, int64_t);
PRE_DECLARATION(tstring, int8);
PRE_DECLARATION(tstring, Eigen::half);
PRE_DECLARATION(tstring, bfloat16)

#undef PRE_DECLARATION


}   // namespace op_registry
}   // namespace recommenders_addons
}   // namespace tensorflow