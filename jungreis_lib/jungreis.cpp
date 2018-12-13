#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

static std::vector<std::vector<double>> cacheData;

extern "C" __declspec(dllexport) void reserveCache(size_t n, size_t m) {

    std::vector<double> row(m, NAN);

    cacheData = std::vector<std::vector<double>>(n, row);
}

double getCacheValue(size_t n, size_t m)
{
    if (n >= cacheData.size())
        return NAN;
    if (m >= cacheData[n].size())
        return NAN;
    return cacheData[n][m];
}

void setCacheValue(size_t n, size_t m, double value)
{
    if (n >= cacheData.size())
    {
        cacheData.insert(cacheData.end(), n - cacheData.size() + 1, std::vector<double>());
    }

    auto& row = cacheData[n];
    if (m >= row.size())
    {
        row.insert(row.end(), m - row.size() + 1, NAN);
    }
    
    row[m] = value;
}

extern "C" __declspec(dllexport) double betaF(int64_t n, int64_t m) {
    const int64_t nnn = (1 << (n + 1)) - 1;


    const double cachedValue = getCacheValue(n, m);
    if (!isnan(cachedValue))
    {
        return cachedValue;
    }

    if (m == 0)
        return 1.0;

    if (n > 0 && m < nnn)
        return 0.0;

    double value = 0;
    for (int64_t k = nnn; k < m - nnn + 1; ++k)
    {
        value += betaF(n, k) * betaF(n, m - k);
    }

    value = (betaF(n + 1, m) - value - betaF(0, m - nnn)) / 2.0;

    if (isnan(cachedValue))
        setCacheValue(n, m, value);

    return value;
}

extern "C" __declspec(dllexport) void printCacheUsage()
{

    size_t maxSize = 0;
    for (size_t n = 0; n < cacheData.size(); ++n)
    {
        maxSize = std::max(maxSize, cacheData[n].size());
    }

    std::cout << "# rows: " << cacheData.size() << " longest row: " << maxSize << std::endl;
}