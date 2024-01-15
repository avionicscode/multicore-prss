#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_set>
#include <array>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <set>
#include <cmath>
#include <algorithm>
#include <omp.h>

#define SLICE 32

class skyline
{
    friend auto operator<<(std::ostream &, const skyline &) -> std::ostream &;

public:
    // size_t DT;
    skyline() = default;
    virtual ~skyline() = default;
    // Add a skyline point to d-tree.
    auto add(const size_t &) -> skyline &;
    // The first skyline point dominates the second point.
    auto append(const size_t &, const size_t &) -> skyline &;
    // Check whether a point is skyline point.
    auto contains(const size_t &) -> bool;
    // Get all dominated points of a skyline point.
    auto get(const size_t &) -> std::vector<size_t> &;
    // Move the first skyline point to the second skyline point.
    auto move(const size_t &, const size_t &) -> skyline &;
    // Remove a skyline point.
    auto remove(const size_t &) -> skyline &;
    // The number of skyline points.
    auto size() -> size_t;

private:
    inline auto slice_(size_t) -> size_t;
    size_t count_ = 0;
    std::vector<size_t> empty_;
    std::array<std::unordered_map<size_t, std::vector<size_t>>, SLICE> tree_;
};

auto dominate(const double *row1, const double *row2, size_t width) -> bool
{
    //++skyline::DT;
    const double *p1 = row1;
    const double *p2 = row2;
    bool dominating = false;
    for (size_t i = 0; i < width; ++i, ++p1, ++p2)
    {
        if (*p1 > *p2)
        {
            return false;
        }
        else if (*p1 < *p2 && !dominating)
        {
            dominating = true;
        }
    }
    return dominating;
}

// Function to check if a point V2LessTV1 another point
bool V2LessTV1(const std::vector<double> &p1, const std::vector<double> &p2)
{
    bool dominating = false;
    for (size_t i = 0; i < p1.size(); ++i)
    {
        if (p1[i] > p2[i])
        {
            return false;
        }
        else if (p1[i] < p2[i] && !dominating)
        {
            dominating = true;
        }
    }
    return dominating;
}

// size_t skyline::DT = 0;

auto operator<<(std::ostream &out, const skyline &skyline) -> std::ostream &
{
    for (const auto &tree : skyline.tree_)
    {
        for (const auto &s : tree)
        {
            out << "Key: " << s.first << " Values: ";
            for (const auto &value : s.second)
            {
                out << value << " ";
            }
            out << std::endl;
        }
    }
    return out;
}

auto skyline::add(const size_t &s) -> skyline &
{
    auto &&it = tree_[slice_(s)].find(s);
    if (it != tree_[slice_(s)].end())
    {
        return *this;
    }
    std::vector<size_t> v;
    tree_[slice_(s)].insert(std::make_pair(s, v));
    ++count_;
    return *this;
}

auto skyline::append(const size_t &s, const size_t &p) -> skyline &
{
    auto &&it = tree_[slice_(s)].find(s);
    if (it == tree_[slice_(s)].end())
    {
        std::vector<size_t> v;
        v.push_back(p);
        tree_[slice_(s)].insert(std::make_pair(s, v));
        return *this;
    }
    it->second.push_back(p);
    return *this;
}

auto skyline::contains(const size_t &p) -> bool
{
    return tree_[slice_(p)].count(p);
}

auto skyline::get(const size_t &s) -> std::vector<size_t> &
{
    auto &&it = tree_[slice_(s)].find(s);
    if (it == tree_[slice_(s)].end())
    {
        return empty_;
    }
    return it->second;
}

auto skyline::move(const size_t &s1, const size_t &s2) -> skyline &
{
    auto &&it1 = tree_[slice_(s1)].find(s1);
    if (it1 == tree_[slice_(s1)].end())
    {
        return *this;
    }
    auto &&it2 = tree_[slice_(s2)].find(s2);
    if (it2 == tree_[slice_(s2)].end())
    {
        return *this;
    }
    it2->second.push_back(s1);
    it2->second.insert(it2->second.end(), it1->second.begin(), it1->second.end());
    tree_[slice_(s1)].erase(it1);
    --count_;
    return *this;
}

auto skyline::remove(const size_t &s) -> skyline &
{
    tree_[slice_(s)].erase(s);
    --count_;
    return *this;
}

auto skyline::size() -> size_t
{
    return count_;
}

auto skyline::slice_(size_t index) -> size_t
{
    return (int)index % SLICE;
}

struct cache_entry
{
    size_t index = 0;
    double value = 0;
    explicit cache_entry(size_t);
    cache_entry(size_t, double);
    cache_entry() = default;
};

cache_entry::cache_entry(size_t index) : index(index)
{
}

cache_entry::cache_entry(size_t index, double value) : index(index), value(value)
{
}

auto operator==(const cache_entry &row1, const cache_entry &row2) -> bool
{
    return row1.index == row2.index;
}

auto operator<(const cache_entry &row1, const cache_entry &row2) -> bool
{
    if (row1.value == row2.value)
    {
        return row1.index < row2.index;
    }
    return row1.value < row2.value;
}

auto operator<<(std::ostream &out, const cache_entry &row) -> std::ostream &
{
    out << row.value << ":" << row.index;
    return out;
}

// ent: entries dim: dimension
auto estimate(const cache_entry &ent, const std::set<cache_entry> &dim) -> double
{
    if (dim.empty())
    {
        return 0;
    }
    auto &&first = *(dim.begin());
    auto &&last = *(dim.rbegin());
    if (first == last)
    {
        return 1;
    }
    if (ent < first)
    {
        return 0;
    }
    if (last < ent)
    {
        return 1;
    }
    return 1.0 * std::abs(ent.value - first.value) / std::abs(last.value - first.value);
}

auto lower_dimension(const cache_entry *entries, const std::set<cache_entry> *indexes, size_t width) -> size_t
{
    size_t d = 0;
    double lower = 1;
    for (size_t i = 0; i < width; ++i)
    {
        double est = estimate(entries[i], indexes[i]);
        if (est == 0)
        {
            return i;
        }
        else
        {
            if (est < lower)
            {
                lower = est;
                d = i;
            }
        }
    }
    return d;
}

auto upper_dimension(const cache_entry *entries, const std::set<cache_entry> *indexes, size_t width) -> size_t
{
    size_t d = 0;
    double upper = 0;
    for (size_t i = 0; i < width; ++i)
    {
        double est = estimate(entries[i], indexes[i]);
        if (est == 1)
        {
            return i;
        }
        else
        {
            if (est > upper)
            {
                upper = est;
                d = i;
            }
        }
    }
    return d;
}

class cache
{
public:
    cache() = default;
    cache(size_t, size_t);
    virtual ~cache();
    auto get(size_t) -> double *;
    auto put(double *, bool) -> double *;
    auto skyline(size_t) -> bool &;

private:
    double *cache_ = nullptr;
    size_t count_ = 0;
    bool *skyline_ = nullptr;
    size_t width_ = 0;
    size_t window_ = 0;
};

// -----------------------------------------------------------------------------------
cache::cache(size_t width, size_t window) : count_(0), width_(width), window_(window)
{
    cache_ = new double[width_ * window_];
    skyline_ = new bool[window_];
}

cache::~cache()
{
    delete[] cache_;
    delete[] skyline_;
}

auto cache::get(size_t index) -> double *
{
    /*
    if (index >= count_)
    {
        return nullptr;
    }
    */

    return &cache_[(index % window_) * width_];
}

auto cache::put(double *buffer, bool skyline) -> double *
{
    size_t index = count_ % window_;
    double *base = &cache_[index * width_];
    std::memcpy(base, buffer, sizeof(double) * width_);
    skyline_[index] = skyline;
    ++count_;
    return base;
}

auto cache::skyline(size_t index) -> bool &
{
    return skyline_[index % window_];
}

void readTextPointList(int n, int d, const std::string &strFileName, std::vector<std::vector<double>> &result)
{
    std::vector<std::vector<double>> PointList;
    std::string strLine;
    std::ifstream inFile(strFileName);

    if (inFile.is_open())
    {
        int i = 0;
        while (getline(inFile, strLine) && i < n)
        {
            if (strLine.size() != 0)
            {
                std::stringstream sin(strLine);
                std::vector<double> values(d);
                for (int j = 0; j < d; j++)
                {
                    double aa;
                    sin >> aa;
                    values[j] = aa;
                }
                PointList.push_back(values);
                i++;
            }
        }
        inFile.close();

        result = PointList;
    }
    else
    {
        std::cout << "Error: Unable to open file." << std::endl;
    }
}

void skyline_update(std::vector<std::vector<double>> &data, size_t width, size_t window, size_t start, size_t end, std::unordered_set<size_t> &deal, cache_entry *entries, cache_entry *entries_update, cache_entry *entries_remove, std::set<cache_entry> *indexes, double *tuple, skyline &Skyline, cache &cache, std::vector<std::vector<double>> &localSkyline)
{
    size_t index = start; // Index ID of the incoming tuple.

    for (size_t i = 0; i < width; ++i)
    {
        tuple[i] = data[start][i];
        // tuple[i] = data[0][i];
        entries[i].index = index;
        entries[i].value = data[start][i];
        // entries[i].value = data[0][i];
        indexes[i].insert(entries[i]);
    }

    cache.put(tuple, true);
    Skyline.add(index);
    ++index;

    // Process incoming tuples.
    // for (size_t j = 1; j < data.size(); ++j)

    for (size_t j = (start + 1); j < end; ++j)
    {
        for (size_t i = 0; i < width; ++i)
        {
            tuple[i] = data[j][i];
            entries[i].index = index;
            entries[i].value = data[j][i];
            // entries[i].value = tuple[i];
        }

        if (index >= end)
        {
            auto &&index_remove = index - window;
            auto &&tuple_remove = cache.get(index_remove);
            for (size_t i = 0; i < width; ++i)
            {
                entries_remove[i].index = index_remove;
                entries_remove[i].value = tuple_remove[i];
            }

            if (cache.skyline(index_remove))
            {
                deal.clear();
                for (auto &&index_update : Skyline.get(index_remove))
                {

                    if (index_update < index_remove)
                    {
                        continue;
                    }
                    deal.insert(index_update);
                    auto &&tuple_update = cache.get(index_update);
                    for (size_t i = 0; i < width; ++i)
                    {
                        entries_update[i].index = index_update;
                        entries_update[i].value = tuple_update[i];
                    }
                    auto &&lower_bound_dimension = lower_dimension(entries_update, indexes, width);
                    auto &&lower_bound_entry = entries_update[lower_bound_dimension];
                    auto &&lower_bound_index = indexes[lower_bound_dimension];
                    auto &&lower = lower_bound_index.begin();
                    bool dominated = false;

                    while (lower->value <= lower_bound_entry.value && lower != lower_bound_index.end())
                    {
                        if (!cache.skyline(lower->index) || lower->index == index_remove)
                        {
                            ++lower;
                            continue;
                        }

                        if (dominate(cache.get(lower->index), tuple_update, width))
                        {
                            Skyline.append(lower->index, index_update);
                            dominated = true;
                            break;
                        }

                        ++lower;
                    }
                    if (!dominated)
                    {
                        cache.skyline(index_update) = true;
                        Skyline.add(index_update);
                    }

                    for (auto &&x : deal)
                    {
                        if (x != index_update && cache.skyline(x))
                        {
                            if (dominate(tuple_update, cache.get(x), width))
                            {
                                cache.skyline(x) = false;
                                Skyline.move(x, index_update);
                            }
                        }
                    }
                }
                cache.skyline(index_remove) = false;
                Skyline.remove(index_remove);
            }

            for (size_t i = 0; i < width; ++i)
            {
                indexes[i].erase(entries_remove[i]);
            }
        }

        // Do lower-bound dominance checking.
        bool dominated = false;
        auto &&lower_bound_dimension = lower_dimension(entries, indexes, width);
        auto &&lower_bound_entry = entries[lower_bound_dimension];
        auto &&lower_bound_index = indexes[lower_bound_dimension];
        auto &&lower = lower_bound_index.begin();
        while (lower->value <= lower_bound_entry.value && lower != lower_bound_index.end())
        {
            // Only compare the incoming tuple with skyline tuples.
            if (!cache.skyline(lower->index))
            {
                ++lower;
                continue;
            }
            // If the incoming tuple is dominated by a lower skyline tuple, do break.
            // The skyline flag of the incoming tuple will be set while adding it
            // to the cache.
            if (dominate(cache.get(lower->index), tuple, width))
            {
                dominated = true;
                Skyline.append(lower->index, index);
                break;
            }
            // If the incoming tuple is not dominated by the lower tuple, however the
            // lower tuple has the same value as the current tuple, then do reverse
            // dominance checking.
            if (lower->value == lower_bound_entry.value && dominate(tuple, cache.get(lower->index), width))
            {
                cache.skyline(lower->index) = false;
                Skyline.move(lower->index, index);
            }
            ++lower;
        }
        // Do upper-bound dominance checking.
        if (!dominated)
        {
            Skyline.add(index);
            auto &&upper_bound_dimension = upper_dimension(entries, indexes, width);
            auto &&upper_bound_entry = entries[upper_bound_dimension];
            auto &&upper_bound_index = indexes[upper_bound_dimension];
            auto &&upper_repeat = std::set<cache_entry>::reverse_iterator(upper_bound_index.lower_bound(upper_bound_entry));
            // For repeating dimensional values.
            while (upper_repeat != upper_bound_index.rend())
            {
                if (!cache.skyline(upper_repeat->index))
                {
                    ++upper_repeat;
                    continue;
                }
                if (upper_repeat->value < upper_bound_entry.value)
                {
                    break;
                }
                // A tuple with repeat dimensional value is dominated by the incoming
                // tuple.
                if (dominate(tuple, cache.get(upper_repeat->index), width))
                {
                    cache.skyline(upper_repeat->index) = false;
                    Skyline.move(upper_repeat->index, index);
                }
                ++upper_repeat;
            }
            // Find all upper skyline tuples that are dominated by the
            // incoming tuple.
            auto &&upper = upper_bound_index.upper_bound(upper_bound_entry);

            while (upper != upper_bound_index.end())
            {
                if (!Skyline.contains(upper->index))
                {
                    // if (!cache.skyline(upper->index)) {
                    ++upper;
                    continue;
                }
                if (dominate(tuple, cache.get(upper->index), width))
                {
                    cache.skyline(upper->index) = false;
                    Skyline.move(upper->index, index);
                }
                ++upper;
            }
        }
        // Add the incoming tuple to all dimensional indexes.
        for (size_t i = 0; i < width; ++i)
        {
            indexes[i].insert(entries[i]);
        }

        // Finally, replace the expired tuple by the incoming tuple.
        cache.put(tuple, !dominated);

        ++index;
    }

    // id = start; id < end; ++id
    for (size_t id = 0; id < window; ++id)
    {
        if (cache.skyline(id))
        {
            std::vector<double> row;

            for (size_t i = 0; i < width; ++i)
            {
                row.push_back(cache.get(id)[i]);
            }

            localSkyline.push_back(row);
        }
    }
}

// Function to combine local skylines into the global skyline
std::vector<std::vector<double>> combineSkylines(const std::vector<std::vector<double>> &localSkylines)
{
    std::vector<std::vector<double>> globalSkyline;

    for (const std::vector<double> &p : localSkylines)
    {
        bool isDominated = false;

        auto it = globalSkyline.begin();
        while (it != globalSkyline.end())
        {
            if (dominate(p.data(), it->data(), p.size()))
            {
                isDominated = true;
                break;
            }
            else if (dominate(it->data(), p.data(), p.size()))
            {
                // If the current point in the global skyline dominates the new point, remove it.
                it = globalSkyline.erase(it);
            }
            else
            {
                // Move to the next point in the global skyline.
                ++it;
            }
        }

        if (!isDominated)
        {
            globalSkyline.push_back(p);
        }
    }

    return globalSkyline;
}

auto main(int argc, char **argv) -> int
{

    if (argc < 2)
    {
        std::cout << "Usage: DIMENSIONALITY WINDOW" << std::endl;
        return 0;
    }

    size_t dimensionality = strtoul(argv[1], nullptr, 10);
    size_t window = strtoul(argv[2], nullptr, 10);

    // Read the data into a shared vector
    // std::string filename = "small.txt";
    std::string filename = "elv_weather-U-15-401000.txt";
    size_t datasize = 401000;

    std::vector<std::vector<double>> data;
    readTextPointList(datasize, dimensionality, filename, data);

    std::cout << "========== start ==========" << std::endl;

    const size_t core_number = 12;
    size_t step = datasize / core_number + 1;

    std::vector<std::vector<double>> localSkylines;

    double startTime = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
    for (int i = core_number - 1; i >= 0; i--)
    {
        skyline Skyline;

        std::vector<std::vector<double>> localSkyline[core_number];
        // std::vector<std::vector<double>> localSkyline;
        std::unordered_set<size_t> deal;
        auto entries = new cache_entry[dimensionality];
        auto entries_remove = new cache_entry[dimensionality];
        auto entries_update = new cache_entry[dimensionality];
        auto indexes = new std::set<cache_entry>[dimensionality];
        auto tuple = new double[dimensionality];

        cache cache(dimensionality, window); // Tuple cache.

        skyline_update(data, dimensionality, window, i * step, std::min((unsigned int)((i + 1) * step), (unsigned int)window), deal, entries, entries_update, entries_remove, indexes, tuple, Skyline, cache, localSkyline[i]);

        // Combine local skylines outside the parallel region
#pragma omp critical
        {
            localSkylines.insert(localSkylines.end(), localSkyline[i].begin(), localSkyline[i].end());
        }

        delete[] entries;
        delete[] entries_remove;
        delete[] entries_update;
        delete[] indexes;
        delete[] tuple;
    }

    std::vector<std::vector<double>> globalSkyline = combineSkylines(localSkylines);

    double stopTime = omp_get_wtime();
    double secsElapsed = stopTime - startTime;
    std::cout << "parallel Time Skyline: " << secsElapsed << std::endl;

    std::cout << "========== end ==========" << std::endl;

    return 0;
}

// cl.exe /EHsc /O2 /openmp:llvm /MD /W4 /WX /fsanitize=address /Zi /std:c++20 Finallsolution.cpp