#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
#include <list>
#include <limits>

namespace py = pybind11;

enum class Order
{
    ROW_MAJOR,
    COLUMN_MAJOR,
    UNKNOWN
};

Order check_order(const py::buffer_info &info)
{
    size_t itemsize = info.itemsize;
    size_t strides0 = info.strides[0];
    size_t strides1 = info.strides[1];

    if (strides0 == itemsize && strides1 == itemsize * info.shape[0])
    {
        return Order::COLUMN_MAJOR;
    }
    else if (strides0 == itemsize * info.shape[1] && strides1 == itemsize)
    {
        return Order::ROW_MAJOR;
    }
    else
    {
        return Order::UNKNOWN;
    }
}

// 输出dataframe中的元素
template <typename T>
void _print_mat(const py::array_t<T> &df_array)
{
    py::buffer_info info = df_array.request();
    T *ptr = static_cast<T *>(info.ptr);
    std::size_t rows = info.shape[0];
    std::size_t cols = info.shape[1];

    // 检查内存布局
    size_t itemsize = info.itemsize;
    size_t strides0 = info.strides[0];
    size_t strides1 = info.strides[1];

    py::print("itemsize", itemsize, "strides0", strides0, "strides1", strides1);
    if (strides0 == itemsize && strides1 == itemsize * rows)
    {
        py::print("column-major order");
    }
    else if (strides0 == itemsize * cols && strides1 == itemsize)
    {
        py::print("row-major order");
    }
    else
    {
        py::print("unknown order");
    }

    py::print("rows", rows, "cols", cols);
    // for (int i = 0; i < rows * cols; i++)
    // {
    //     py::print(ptr[i]);
    // }
}

void print_mat_float(const py::array_t<float> &df_array)
{
    _print_mat<float>(df_array);
}

void print_mat(const py::array_t<double> &df_array)
{
    _print_mat<double>(df_array);
}

// 直接使用列表计算方差
// 内存占用大
template <typename T>
std::tuple<T, T> _cal_std(std::list<T> &v, T sum)
{

    // for (const auto &i : v)
    // {
    //     py::print(i);
    // }
    // py::print("len(v)", v.size(), "sum", sum);
    // py::print("");

    T mean = 0.0;
    T std_dev = 0.0;

    if (!v.empty())
    {
        mean = sum / v.size();

        T sum_squares = 0.0;
        T diff = 0.0;
        for (T value : v)
        {
            diff = value - mean;
            sum_squares += diff * diff;
        }
        std_dev = std::sqrt(sum_squares / v.size());
    }

    return std::make_tuple(mean, std_dev);
}

const float acc = 1000.0;
// 使用map缩放来计算方差
// 精度损失
template <typename T>
std::tuple<T, T> _cal_std(std::map<int, size_t> &v, T sum, size_t size)
{
    T mean = 0.0;
    T std_dev = 0.0;

    if (!v.empty())
    {
        mean = sum / size;

        T sum_squares = 0.0;
        T diff = 0.0;
        for (auto [value, _size] : v)
        {
            diff = value / acc - mean;
            sum_squares += diff * diff * _size;

            // if (std::isnan(std::sqrt(sum_squares / size)))
            // {
            //     py::print("value:", value, "size:", _size, "sum_squares:", sum_squares, "size:", size);
            //     break;
            // }
        }
        std_dev = std::sqrt(sum_squares / size);
    }

    return std::make_tuple(mean, std_dev);
}

// 使用map储存数据
// 计算输入价格数据的均值方差
// 使用mid_price输入 作为 midprice
// 所有数据调整成与 midprice 的数值，根据 func
// 按列顺序返回 均值方差
template <typename T>
std::vector<std::tuple<T, T>> _cal_price_mean_std_each_col_map(const py::array_t<T> &df_array, const py::array_t<T> &mid_price, int pass_n, std::function<T(const T &, const T &)> func)
{
    py::buffer_info info = df_array.request();
    T *ptr = static_cast<T *>(info.ptr);
    std::size_t rows = info.shape[0];
    std::size_t cols = info.shape[1];

    py::buffer_info info_mid_price = mid_price.request();
    T *ptr_mid_price = static_cast<T *>(info_mid_price.ptr);

    py::print("rows", rows, "cols", cols);

    std::vector<std::tuple<T, T>> res;

    std::map<int, size_t> v;

    T sum = 0.0;
    size_t size = 0;
    size_t __idx = 0;
    size_t _idx = 0;
    T base_price;
    T key_v_d;
    int key_v;
    for (int col_idx = 0; col_idx < cols; col_idx++)
    {
        v.clear();
        sum = 0.0;
        size = 0;

        __idx = col_idx * rows;
        for (int i = pass_n - 1; i < rows; i++)
        {
            base_price = ptr_mid_price[i];
            py::print("mid_price:", base_price);

            _idx = __idx + i;
            for (int row_idx = 0; row_idx < pass_n; row_idx++)
            {
                key_v_d = func(ptr[_idx - row_idx], base_price);
                py::print(ptr[_idx - row_idx], base_price, key_v_d);

                key_v = (int)(key_v_d * acc);

                v[key_v]++;
                sum += key_v_d;
                size++;

                if (size < 0)
                {
                    py::print("size:", size);
                    // return std::make_tuple(0.0, 0.0);
                }
            }
        }

        res.push_back(_cal_std<T>(v, sum, size));
    }

    return res;
}

// 使用list 储存数据
// 计算输入价格数据的均值方差
// 使用mid_price输入 作为 midprice
// 所有数据调整成与 midprice 的数值，根据 func
// 按列顺序返回 均值方差
template <typename T>
std::vector<std::tuple<T, T>> _cal_price_mean_std_each_col(const py::array_t<T> &df_array, const py::array_t<T> &mid_price, int pass_n, std::function<T(const T &, const T &)> func)
{
    py::buffer_info info = df_array.request();
    T *ptr = static_cast<T *>(info.ptr);
    std::size_t rows = info.shape[0];
    std::size_t cols = info.shape[1];

    py::buffer_info info_mid_price = mid_price.request();
    T *ptr_mid_price = static_cast<T *>(info_mid_price.ptr);

    // py::print("rows", rows, "cols", cols);
    Order order = check_order(info);
    if (order == Order::UNKNOWN)
    {
        py::print("unknown order");
        std::abort();
    }

    std::vector<std::tuple<T, T>> res;

    std::list<T> v;

    T sum = 0.0;
    size_t size = 0;
    size_t __idx = 0;
    size_t _idx = 0;
    T base_price;
    for (size_t col_idx = 0; col_idx < cols; col_idx++)
    {
        v.clear();
        sum = 0.0;

        if (order == Order::ROW_MAJOR)
        {
            for (size_t i = pass_n - 1; i < rows; i++)
            {
                _idx = i * cols + col_idx;

                base_price = ptr_mid_price[i];

                for (int row_idx = 0; row_idx < pass_n; row_idx++)
                {
                    // py::print(_idx, ptr[_idx], base_price);
                    v.push_back(func(ptr[_idx], base_price));
                    sum += v.back();
                    _idx -= cols;
                }
            }
        }
        else
        {
            __idx = col_idx * rows;

            for (size_t i = pass_n - 1; i < rows; i++)
            {
                _idx = __idx + i;

                base_price = ptr_mid_price[i];

                for (int row_idx = 0; row_idx < pass_n; row_idx++)
                {
                    v.push_back(func(ptr[_idx - row_idx], base_price));
                    sum += v.back();
                }
            }
        }

        res.push_back(_cal_std<T>(v, sum));
    }

    return res;
}

template <typename T>
struct array_info
{
    std::size_t rows;
    std::size_t cols;
    T *ptr;
    Order order;
};

// 使用list 储存数据
// 计算输入价格数据的均值方差
// 使用mid_price输入 作为 midprice
// 所有数据调整成与 midprice 的数值，根据 func
// 按列顺序返回 均值方差
template <typename T>
std::vector<std::tuple<T, T>> _cal_price_mean_std_each_col_multi(const std::vector<py::array_t<T>> &df_array, const std::vector<py::array_t<T>> &mid_price, int pass_n, std::function<T(const T &, const T &)> func, const std::vector<py::array_t<T>> &time_diff_max={})
{
    std::vector<array_info<T>> ptrs;
    for (auto &df : df_array)
    {
        py::buffer_info info = df.request();
        T *ptr = static_cast<T *>(info.ptr);
        std::size_t rows = info.shape[0];
        std::size_t cols = info.shape[1];

        // py::print("rows", rows, "cols", cols);
        Order order = check_order(info);
        if (order == Order::UNKNOWN)
        {
            py::print("unknown order");
            std::abort();
        }

        array_info<T> array_obj;
        array_obj.cols = cols;
        array_obj.rows = rows;
        array_obj.ptr = ptr;
        array_obj.order = order;

        ptrs.push_back(array_obj);
    }

    std::vector<T *> mid_price_ptrs;
    for (auto &df : mid_price)
    {
        py::buffer_info info = df.request();
        T *ptr = static_cast<T *>(info.ptr);
        mid_price_ptrs.push_back(ptr);
    }

    std::vector<T *> time_diff_max_ptrs;
    bool use_time_diff_max = time_diff_max.size() > 0;
    if (use_time_diff_max){
        for (auto &df : time_diff_max)
        {
            py::buffer_info info = df.request();
            T *ptr = static_cast<T *>(info.ptr);
            time_diff_max_ptrs.push_back(ptr);
        }
    }

    std::vector<std::tuple<T, T>> res;

    std::list<T> v;

    T sum = 0.0;
    size_t size = 0;
    size_t __idx = 0;
    size_t _idx = 0;
    T base_price;
    T time_diff;
    const std::size_t cols = ptrs[0].cols;
    for (size_t col_idx = 0; col_idx < cols; col_idx++)
    {
        v.clear();
        sum = 0.0;

        int idx = 0;
        for (auto &array_obj : ptrs)
        {
            const std::size_t rows = array_obj.rows;
            T *ptr = array_obj.ptr;
            const Order order = array_obj.order;
            T *ptr_mid_price = mid_price_ptrs[idx];

            T *ptr_time_diff_max = nullptr;
            if (use_time_diff_max)
                T *ptr_time_diff_max = time_diff_max_ptrs[idx];
            idx++;

            if (order == Order::ROW_MAJOR)
            {
                for (size_t i = pass_n - 1; i < rows; i++)
                {
                    _idx = i * cols + col_idx;

                    base_price = ptr_mid_price[i];

                    // 确保 time_diff 满足条件
                    if (use_time_diff_max){
                        time_diff = ptr_time_diff_max[i];
                        if (time_diff == 0)
                            continue;
                    }

                    for (int row_idx = 0; row_idx < pass_n; row_idx++)
                    {
                        T &_v = ptr[_idx];
                        if (!std::isnan(_v) && !std::isinf(_v))
                        {
                            v.push_back(func(_v, base_price));
                            sum += v.back();
                        }

                        _idx -= cols;
                    }
                }
            }
            else
            {
                __idx = col_idx * rows;

                for (size_t i = pass_n - 1; i < rows; i++)
                {
                    _idx = __idx + i;

                    base_price = ptr_mid_price[i];

                    // 确保 time_diff 满足条件
                    if (use_time_diff_max){

                        time_diff = ptr_time_diff_max[i];
                        if (time_diff == 0)
                            continue;
                    }

                    for (int row_idx = 0; row_idx < pass_n; row_idx++)
                    {
                        T &_v = ptr[_idx - row_idx];
                        if (!std::isnan(_v) && !std::isinf(_v))
                        {
                            v.push_back(func(_v, base_price));
                            sum += v.back();
                        }
                    }
                }
            }
        }

        res.push_back(_cal_std<T>(v, sum));
    }

    return res;
}

// 使用map储存数据
// 计算输入数据的均值方差
// 按列顺序返回 均值方差
template <typename T>
std::vector<std::tuple<T, T>> _cal_mean_std_each_col_map(const py::array_t<T> &df_array, int pass_n)
{
    py::buffer_info info = df_array.request();
    T *ptr = static_cast<T *>(info.ptr);
    std::size_t rows = info.shape[0];
    std::size_t cols = info.shape[1];

    py::print("rows", rows, "cols", cols);

    std::vector<std::tuple<T, T>> res;

    std::map<int, size_t> v;

    T sum = 0.0;
    size_t size = 0;
    size_t __idx = 0;
    size_t _idx = 0;
    int key_v;
    for (int col_idx = 0; col_idx < cols; col_idx++)
    {
        v.clear();
        sum = 0.0;
        size = 0;

        __idx = col_idx * rows;
        for (int i = pass_n - 1; i < rows; i++)
        {
            _idx = __idx + i;
            for (int row_idx = 0; row_idx < pass_n; row_idx++)
            {
                key_v = (int)(ptr[_idx - row_idx] * acc);

                v[key_v]++;
                sum += ptr[_idx - row_idx];
                size++;

                if (size < 0)
                {
                    py::print("size:", size);
                    // return std::make_tuple(0.0, 0.0);
                }
            }
        }

        res.push_back(_cal_std(v, sum, size));
    }

    return res;
}

// 使用list 储存数据
// 计算输入数据的均值方差
// 按列顺序返回 均值方差
template <typename T>
std::vector<std::tuple<T, T>> _cal_mean_std_each_col(const py::array_t<T> &df_array, int pass_n)
{
    py::buffer_info info = df_array.request();
    T *ptr = static_cast<T *>(info.ptr);
    const std::size_t rows = info.shape[0];
    const std::size_t cols = info.shape[1];

    // py::print("rows", rows, "cols", cols);
    Order order = check_order(info);
    if (order == Order::UNKNOWN)
    {
        py::print("unknown order");
        std::abort();
    }

    std::vector<std::tuple<T, T>> res;

    std::list<T> v;

    T sum = 0.0;
    size_t _idx = 0;
    size_t __idx = 0;
    for (size_t col_idx = 0; col_idx < cols; col_idx++)
    {
        v.clear();
        sum = 0.0;

        if (order == Order::ROW_MAJOR)
        {
            for (size_t i = pass_n - 1; i < rows; i++)
            {
                _idx = i * cols + col_idx;
                for (int row_idx = 0; row_idx < pass_n; row_idx++)
                {
                    v.push_back(ptr[_idx]);
                    sum += v.back();

                    _idx -= cols;
                }
            }
        }
        else
        {
            __idx = col_idx * rows;

            for (size_t i = pass_n - 1; i < rows; i++)
            {
                _idx = __idx + i;
                for (int row_idx = 0; row_idx < pass_n; row_idx++)
                {
                    v.push_back(ptr[_idx - row_idx]);
                    sum += v.back();
                }
            }
        }

        res.push_back(_cal_std(v, sum));
    }

    return res;
}

// 使用list 储存数据
// 计算输入数据的均值方差
// 按列顺序返回 均值方差
template <typename T>
std::vector<std::tuple<T, T>> _cal_mean_std_each_col_multi(const std::vector<py::array_t<T>> &df_array, int pass_n, const std::vector<py::array_t<T>> &time_diff_max={})
{
    std::vector<array_info<T>> ptrs;
    for (auto &df : df_array)
    {
        py::buffer_info info = df.request();
        T *ptr = static_cast<T *>(info.ptr);
        std::size_t rows = info.shape[0];
        std::size_t cols = info.shape[1];

        // py::print("rows", rows, "cols", cols);
        Order order = check_order(info);
        if (order == Order::UNKNOWN)
        {
            py::print("unknown order");
            std::abort();
        }

        array_info<T> array_obj;
        array_obj.cols = cols;
        array_obj.rows = rows;
        array_obj.ptr = ptr;
        array_obj.order = order;

        ptrs.push_back(array_obj);
    }

    std::vector<T *> time_diff_max_ptrs;
    bool use_time_diff_max = time_diff_max.size() > 0;
    if (use_time_diff_max){
        for (auto &df : time_diff_max)
        {
            py::buffer_info info = df.request();
            T *ptr = static_cast<T *>(info.ptr);
            time_diff_max_ptrs.push_back(ptr);
        }
    }

    std::vector<std::tuple<T, T>> res;

    std::list<T> v;

    T sum = 0.0;
    T time_diff;
    size_t _idx = 0;
    size_t __idx = 0;
    const std::size_t cols = ptrs[0].cols;
    for (size_t col_idx = 0; col_idx < cols; col_idx++)
    {
        v.clear();
        sum = 0.0;

        int idx = 0;
        for (auto &array_obj : ptrs)
        {
            const std::size_t rows = array_obj.rows;
            T *ptr = array_obj.ptr;
            const Order order = array_obj.order;

            T *ptr_time_diff_max=nullptr;
            if (use_time_diff_max)
                ptr_time_diff_max = time_diff_max_ptrs[idx++];

            if (order == Order::ROW_MAJOR)
            {
                for (size_t i = pass_n - 1; i < rows; i++)
                {
                    _idx = i * cols + col_idx;

                    // 确保 time_diff 满足条件
                    if (use_time_diff_max)
                    {
                        time_diff = ptr_time_diff_max[i];
                        if (time_diff == 0)
                            continue;
                    }

                    for (int row_idx = 0; row_idx < pass_n; row_idx++)
                    {
                        T &_v = ptr[_idx];
                        if (!std::isnan(_v) && !std::isinf(_v))
                        {
                            v.push_back(_v);
                            sum += v.back();
                        }

                        _idx -= cols;
                    }
                }
            }
            else
            {
                __idx = col_idx * rows;

                for (size_t i = pass_n - 1; i < rows; i++)
                {
                    _idx = __idx + i;

                    // 确保 time_diff 满足条件
                    if (use_time_diff_max)
                    {
                        time_diff = ptr_time_diff_max[i];
                        if (time_diff == 0)
                            continue;
                    }

                    for (int row_idx = 0; row_idx < pass_n; row_idx++)
                    {
                        T &_v = ptr[_idx - row_idx];
                        if (!std::isnan(_v) && !std::isinf(_v))
                        {
                            v.push_back(_v);
                            sum += v.back();
                        }
                    }
                }
            }
        }

        res.push_back(_cal_std(v, sum));
    }

    return res;
}

std::vector<std::tuple<float, float>> cal_mean_std_each_col_float(const py::array_t<float> &df_array, int pass_n)
{
    return _cal_mean_std_each_col<float>(df_array, pass_n);
}

std::vector<std::tuple<double, double>> cal_mean_std_each_col(const py::array_t<double> &df_array, int pass_n)
{
    return _cal_mean_std_each_col<double>(df_array, pass_n);
}

std::vector<std::tuple<double, double>> cal_mean_std_each_col_multi(const std::vector<py::array_t<double>> &df_array, int pass_n, const std::vector<py::array_t<double>> &time_diff_max={})
{
    return _cal_mean_std_each_col_multi<double>(df_array, pass_n, time_diff_max);
}

std::vector<std::tuple<double, double>> cal_price_mean_std_pct_each_col(const py::array_t<double> &df_array, const py::array_t<double> &mid_price, int pass_n)
{
    return _cal_price_mean_std_each_col<double>(
        df_array,
        mid_price,
        pass_n,
        [](const double &a, const double &b) -> double
        { return a / b; });
}

std::vector<std::tuple<double, double>> cal_price_mean_std_pct_each_col_multi(const std::vector<py::array_t<double>> &df_array, const std::vector<py::array_t<double>> &mid_price, int pass_n, const std::vector<py::array_t<double>> &time_diff_max={})
{
    return _cal_price_mean_std_each_col_multi<double>(
        df_array,
        mid_price,
        pass_n,
        [](const double &a, const double &b) -> double{ return a / b; },
        time_diff_max
        );
}

std::vector<std::tuple<float, float>> cal_price_mean_std_pct_each_col_float(const py::array_t<float> &df_array, const py::array_t<float> &mid_price, int pass_n)
{
    return _cal_price_mean_std_each_col<float>(
        df_array,
        mid_price,
        pass_n,
        [](const float &a, const float &b) -> float
        { return a / b; });
}

// 计算输入价格数据的均值方差
// 使用mid_price输入 作为 midprice
// 所有数据调整成与 midprice 的差值
std::tuple<double, double> cal_price_mean_std_diff(const py::array_t<double> &df_array, const py::array_t<double> &mid_price, int pass_n)
{
    py::buffer_info info = df_array.request();
    double *ptr = static_cast<double *>(info.ptr);
    std::size_t rows = info.shape[0];
    std::size_t cols = info.shape[1];

    py::buffer_info info_mid_price = mid_price.request();
    double *ptr_mid_price = static_cast<double *>(info_mid_price.ptr);

    py::print("rows", rows, "cols", cols);

    std::map<int, size_t> v;

    double sum = 0.0;
    size_t size = 0;
    size_t _idx = 0;
    double base_price;
    double key_v_d;
    int key_v;

    for (size_t i = pass_n - 1; i < rows; i++)
    {
        base_price = ptr_mid_price[i];
        for (size_t col_idx = 0; col_idx < cols; col_idx++)
        {
            _idx = i + col_idx * rows;
            for (int row_idx = 0; row_idx < pass_n; row_idx++)
            {
                key_v_d = ptr[_idx - row_idx] - base_price;
                // py::print("mid:", base_price, "cur_p", ptr[_idx - row_idx]);
                key_v = (int)(key_v_d * acc);

                v[key_v]++;
                sum += key_v_d;
                size++;

                if (size < 0)
                {
                    py::print("size:", size);
                    return std::make_tuple(0.0, 0.0);
                }
            }
        }
    }

    // return std::make_tuple(0.0, 0.0);
    return _cal_std(v, sum, size);
}

// 计算20列价格数据的均值方差
// 使用(第一列 + 第二列)/2 作为 midprice
// 所有数据调整成与 midprice 的差值
std::tuple<double, double> cal_price_mean_std_diff(const py::array_t<double> &df_array, int pass_n)
{
    py::buffer_info info = df_array.request();
    double *ptr = static_cast<double *>(info.ptr);
    std::size_t rows = info.shape[0];
    std::size_t cols = info.shape[1];

    py::print("rows", rows, "cols", cols);

    std::map<int, size_t> v;

    double sum = 0.0;
    size_t size = 0;
    size_t _idx = 0;
    double base_price;
    double key_v_d;
    int key_v;

    for (size_t i = pass_n - 1; i < rows; i++)
    {
        base_price = (ptr[i] + ptr[i + rows]) / 2;
        for (size_t col_idx = 0; col_idx < cols; col_idx++)
        {
            _idx = i + col_idx * rows;
            for (int row_idx = 0; row_idx < pass_n; row_idx++)
            {
                key_v_d = ptr[_idx - row_idx] - base_price;
                key_v = (int)(key_v_d * acc);

                v[key_v]++;
                sum += key_v_d;
                size++;

                if (size < 0)
                {
                    py::print("size:", size);
                    return std::make_tuple(0.0, 0.0);
                }
            }
        }
    }

    // return std::make_tuple(0.0, 0.0);
    return _cal_std(v, sum, size);
}

void _fillnan_between_nan(const py::array_t<double> &raw, std::vector<int> &col_idxs, std::function<void(double *, const size_t)> func)
{
    py::buffer_info info_raw = raw.request();
    double *ptr_raw = static_cast<double *>(info_raw.ptr);
    std::size_t rows_raw = info_raw.shape[0];
    std::size_t cols_raw = info_raw.shape[1];

    Order order = check_order(info_raw);
    if (order != Order::COLUMN_MAJOR)
    {
        py::print("must be COLUMN_MAJOR order");
        std::abort();
    }

    size_t idx = 0;

    // 构建列索引列表
    if (col_idxs.size() == 0)
    {
        for (size_t i = 0; i < cols_raw; i++)
            col_idxs.push_back(i);
    }

    // 遍历列
    for (auto i : col_idxs)
    {
        idx = i * rows_raw;
        for (size_t j = 0; j < rows_raw; j++)
        {
            func(ptr_raw, idx);
            idx++;
        }

        func(ptr_raw, -1);
    }
}

void fillnan_sum_between_nan(const py::array_t<double> &raw, std::vector<int> col_idxs)
{
    double sum = 0.0;

    auto func = [&sum](double *ptr_raw, const size_t idx)
    {
        if (idx == -1)
        {
            // 重置
            sum = 0.0;
            return;
        }

        if (std::isnan(ptr_raw[idx]))
        {
            ptr_raw[idx] = sum;
            sum = 0.0;
        }
        else
        {
            sum += ptr_raw[idx];
        }
    };

    _fillnan_between_nan(raw, col_idxs, func);
}

void fillnan_count_between_nan(const py::array_t<double> &raw, std::vector<int> col_idxs)
{
    double sum = 0.0;

    auto func = [&sum](double *ptr_raw, const size_t idx)
    {
        if (idx == -1)
        {
            // 重置
            sum = 0.0;
            return;
        }

        if (std::isnan(ptr_raw[idx]))
        {
            ptr_raw[idx] = sum;
            sum = 0.0;
        }
        else
        {
            if (ptr_raw[idx] != 0)
                sum += 1;
        }
    };

    _fillnan_between_nan(raw, col_idxs, func);
}

void fillnan_prod_between_nan(const py::array_t<double> &raw, std::vector<int> col_idxs)
{
    double prod = 1.0;

    auto func = [&prod](double *ptr_raw, const size_t idx)
    {
        if (idx == -1)
        {
            // 重置
            prod = 1.0;
            return;
        }

        if (std::isnan(ptr_raw[idx]))
        {
            ptr_raw[idx] = prod;
            prod = 1.0;
        }
        else
        {
            prod *= ptr_raw[idx];
        }
    };

    _fillnan_between_nan(raw, col_idxs, func);
}

void fillnan_mean_between_nan(const py::array_t<double> &raw, std::vector<int> col_idxs)
{
    double sum = 0.0;
    size_t count = 0;

    auto func = [&](double *ptr_raw, const size_t idx)
    {
        if (idx == -1)
        {
            // 重置
            sum = 0.0;
            count = 0;
            return;
        }

        if (std::isnan(ptr_raw[idx]))
        {
            ptr_raw[idx] = count == 0 ? 0.0 : sum / count;
            sum = 0.0;
            count = 0;
        }
        else
        {
            sum += ptr_raw[idx];
            count++;
        }
    };

    _fillnan_between_nan(raw, col_idxs, func);
}

void test_input_nan_inf(const py::array_t<double> &raw)
{
    py::buffer_info info_raw = raw.request();
    double *ptr_raw = static_cast<double *>(info_raw.ptr);
    std::size_t rows_raw = info_raw.shape[0];
    std::size_t cols_raw = info_raw.shape[1];

    // 遍历输出
    for (size_t i = 0; i < rows_raw; i++)
    {
        for (size_t j = 0; j < cols_raw; j++)
        {
            if (std::isnan(ptr_raw[i * cols_raw + j]))
            {
                py::print("nan", i, j);
            }
            if (std::isinf(ptr_raw[i * cols_raw + j]))
            {
                py::print("inf", i, j);
            }
        }
    }
}

PYBIND11_MODULE(cpp_ext, m)
{
    m.def("cal_price_mean_std_pct_each_col_float", &cal_price_mean_std_pct_each_col_float);
    m.def("cal_price_mean_std_pct_each_col", &cal_price_mean_std_pct_each_col);
    m.def("cal_price_mean_std_pct_each_col_multi", &cal_price_mean_std_pct_each_col_multi);

    m.def("cal_mean_std_each_col_float", &cal_mean_std_each_col_float);
    m.def("cal_mean_std_each_col", &cal_mean_std_each_col);
    m.def("cal_mean_std_each_col_multi", &cal_mean_std_each_col_multi);

    m.def("fillnan_sum_between_nan", &fillnan_sum_between_nan, py::arg("raw"), py::arg("col_idxs") = py::list());
    m.def("fillnan_prod_between_nan", &fillnan_prod_between_nan, py::arg("raw"), py::arg("col_idxs") = py::list());
    m.def("fillnan_count_between_nan", &fillnan_count_between_nan, py::arg("raw"), py::arg("col_idxs") = py::list());
    m.def("fillnan_mean_between_nan", &fillnan_mean_between_nan, py::arg("raw"), py::arg("col_idxs") = py::list());

    // 测试用
    m.def("print_mat_float", &print_mat_float);
    m.def("print_mat", &print_mat);
    m.def("test_input_nan_inf", &test_input_nan_inf);
}