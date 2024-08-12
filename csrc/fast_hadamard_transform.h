/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class OutCastingType {
    out,
    e4m3,
    out_simulated_with_e4m3,
};

#define OUT_CASTING_TYPE_SWITCH(type, ...)                                                              \
    [&] {                                                                                               \
        if (type == OutCastingType::out) {                                                              \
            static constexpr OutCastingType OutCasting = OutCastingType::out;                           \
            return __VA_ARGS__();                                                                       \
        } else if (type == OutCastingType::e4m3) {                                                      \
            static constexpr OutCastingType OutCasting = OutCastingType::e4m3;                          \
            return __VA_ARGS__();                                                                       \
        } else if (type == OutCastingType::out_simulated_with_e4m3) {                                   \
            static constexpr OutCastingType OutCasting = OutCastingType::out_simulated_with_e4m3;       \
            return __VA_ARGS__();                                                                       \
        } else {                                                                                        \
            assert(false and "Unsupported OutCastingType");                                             \
        }                                                                                               \
    }()

////////////////////////////////////////////////////////////////////////////////////////////////////

struct HadamardParamsBase {
    using index_t = int64_t;

    int batch, dim, log_N;

    index_t x_batch_stride;
    index_t out_batch_stride;

    float scale;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ scale_inv_ptr;

    OutCastingType out_casting_type;
};
