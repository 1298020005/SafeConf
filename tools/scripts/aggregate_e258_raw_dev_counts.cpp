// Aggregate Feng UMI counts without converting test-donor perturbation tokens.
// Compile: g++ -O3 -std=c++17 this_file.cpp -lz -o aggregate_e258_raw_dev_counts
#include <zlib.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

static std::vector<std::int32_t> read_mapping(const fs::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open cell-group mapping");
    auto length = input.tellg();
    if (length <= 0 || length % sizeof(std::int32_t) != 0)
        throw std::runtime_error("invalid mapping length");
    std::vector<std::int32_t> mapping(static_cast<size_t>(length) / sizeof(std::int32_t));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(mapping.data()), length);
    if (!input) throw std::runtime_error("cannot read full mapping");
    return mapping;
}

static std::unordered_set<std::string> read_gene_axis(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open gene-axis file");
    std::unordered_set<std::string> genes;
    for (std::string gene; std::getline(input, gene);) {
        if (gene.empty() || gene.find(',') != std::string::npos ||
            gene.find('\r') != std::string::npos)
            throw std::runtime_error("invalid gene-axis entry");
        if (!genes.insert(gene).second)
            throw std::runtime_error("duplicate gene-axis entry");
    }
    if (genes.empty()) throw std::runtime_error("empty gene axis");
    return genes;
}

struct Aggregator {
    const std::vector<std::int32_t>& map;
    const std::unordered_set<std::string>& axis;
    const size_t n_groups;
    std::ofstream matrix;
    std::ofstream genes;
    std::ofstream library;
    std::vector<std::uint32_t> group_sum;
    std::vector<std::uint64_t> group_total;
    std::unordered_set<std::string> seen_raw_genes;
    std::unordered_set<std::string> seen_symbols;
    std::uint64_t selected_tokens = 0;
    std::uint64_t skipped_numeric_tokens = 0;
    std::uint64_t rows_seen = 0;
    std::uint64_t rows_kept = 0;
    bool saw_header = false;

    Aggregator(const std::vector<std::int32_t>& mapping,
               const std::unordered_set<std::string>& selected_axis,
               size_t group_count, const fs::path& binary_path,
               const fs::path& gene_path, const fs::path& library_path)
        : map(mapping), axis(selected_axis), n_groups(group_count),
          matrix(binary_path, std::ios::binary), genes(gene_path),
          library(library_path, std::ios::binary),
          group_sum(group_count, 0), group_total(group_count, 0) {
        if (!matrix || !genes || !library || group_count == 0)
            throw std::runtime_error("cannot create aggregate outputs");
        for (auto group : map)
            if (group < -1 || (group >= 0 && static_cast<size_t>(group) >= n_groups))
                throw std::runtime_error("mapping group index out of range");
    }

    void consume(const std::string& line) {
        if (!saw_header) {
            saw_header = true;
            if (line.empty() || line[0] != ',')
                throw std::runtime_error("raw count header does not start with comma");
            size_t columns = 0;
            for (char c : line) columns += (c == ',');
            if (columns != map.size())
                throw std::runtime_error("header/mapping column count mismatch");
            return;
        }
        ++rows_seen;
        const auto first_comma = line.find(',');
        if (first_comma == std::string::npos || first_comma == 0)
            throw std::runtime_error("invalid gene row or gene symbol");
        const std::string raw_gene = line.substr(0, first_comma);
        const auto first_colon = raw_gene.find(':');
        const auto last_colon = raw_gene.rfind(':');
        const std::string symbol = (first_colon == std::string::npos || last_colon == first_colon)
            ? raw_gene : raw_gene.substr(first_colon + 1, last_colon - first_colon - 1);
        const bool keep = axis.count(symbol) != 0;
        if (keep) {
            if (!seen_raw_genes.insert(raw_gene).second)
                throw std::runtime_error("duplicate raw gene identifier");
            seen_symbols.insert(symbol);  // A symbol may map to multiple raw ENS IDs.
            std::fill(group_sum.begin(), group_sum.end(), 0);
        }
        const char* cursor = line.data() + first_comma + 1;
        const char* end_line = line.data() + line.size();
        for (size_t col = 0; col < map.size(); ++col) {
            const char* delimiter = static_cast<const char*>(
                std::memchr(cursor, ',', static_cast<size_t>(end_line - cursor)));
            const char* end = delimiter ? delimiter : end_line;
            if ((col + 1 < map.size()) != (delimiter != nullptr))
                throw std::runtime_error("gene row column count mismatch");
            const auto group = map[col];
            if (group >= 0) {
                if (cursor == end) throw std::runtime_error("empty selected UMI count");
                std::uint64_t value = 0;
                for (const char* digit = cursor; digit != end; ++digit) {
                    if (*digit < '0' || *digit > '9')
                        throw std::runtime_error("noninteger selected UMI count");
                    value = value * 10 + static_cast<unsigned>(*digit - '0');
                }
                if (value > std::numeric_limits<std::uint64_t>::max() - group_total[group])
                    throw std::runtime_error("group library UMI sum overflow");
                group_total[group] += value;
                if (keep) {
                    if (value > std::numeric_limits<std::uint32_t>::max() - group_sum[group])
                        throw std::runtime_error("group-gene UMI sum overflow");
                    group_sum[group] += static_cast<std::uint32_t>(value);
                }
                ++selected_tokens;
            } else {
                ++skipped_numeric_tokens;  // Delimiters only; no digit conversion.
            }
            cursor = delimiter ? delimiter + 1 : end_line;
        }
        if (keep) {
            matrix.write(reinterpret_cast<const char*>(group_sum.data()),
                         static_cast<std::streamsize>(n_groups * sizeof(std::uint32_t)));
            genes << raw_gene << '\n';
            if (!matrix || !genes) throw std::runtime_error("aggregate output write failed");
            ++rows_kept;
        }
        if (rows_seen % 1000 == 0) {
            std::cerr << "E258 raw rows kept=" << rows_kept
                      << " total gene rows=" << rows_seen << std::endl;
        }
    }

    void write_library() {
        library.write(reinterpret_cast<const char*>(group_total.data()),
                      static_cast<std::streamsize>(n_groups * sizeof(std::uint64_t)));
        if (!library) throw std::runtime_error("group library UMI write failed");
    }
};

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            std::cerr << "usage: aggregate_e258_raw_dev_counts COUNT.gz MAP.i32 "
                         "GENE_AXIS.txt GROUP_COUNT OUTPUT_PREFIX\n";
            return 2;
        }
        const auto mapping = read_mapping(argv[2]);
        const auto axis = read_gene_axis(argv[3]);
        const size_t group_count = std::stoull(argv[4]);
        const fs::path prefix = argv[5];
        const fs::path matrix_final = prefix.string() + ".u32";
        const fs::path genes_final = prefix.string() + ".genes.txt";
        const fs::path library_final = prefix.string() + ".library.u64";
        if (fs::exists(matrix_final) || fs::exists(genes_final) || fs::exists(library_final))
            throw std::runtime_error("aggregate outputs already exist");
        const fs::path matrix_part = matrix_final.string() + ".part";
        const fs::path genes_part = genes_final.string() + ".part";
        const fs::path library_part = library_final.string() + ".part";
        if (fs::exists(matrix_part) || fs::exists(genes_part) || fs::exists(library_part))
            throw std::runtime_error("partial aggregate outputs already exist");
        gzFile input = gzopen(argv[1], "rb");
        if (!input) throw std::runtime_error("cannot open raw count gzip");
        Aggregator aggregator(mapping, axis, group_count, matrix_part, genes_part,
                              library_part);
        std::vector<char> chunk(8 * 1024 * 1024);
        std::string pending;
        while (true) {
            int got = gzread(input, chunk.data(), static_cast<unsigned>(chunk.size()));
            if (got < 0) {
                int status = 0;
                const char* message = gzerror(input, &status);
                throw std::runtime_error(std::string("gzip read failed: ") + message);
            }
            if (got == 0) break;
            const char* cursor = chunk.data();
            const char* end = cursor + got;
            while (cursor != end) {
                const char* nl = static_cast<const char*>(
                    std::memchr(cursor, '\n', static_cast<size_t>(end - cursor)));
                if (!nl) {
                    pending.append(cursor, end);
                    break;
                }
                pending.append(cursor, nl);
                if (!pending.empty() && pending.back() == '\r') pending.pop_back();
                aggregator.consume(pending);
                pending.clear();
                cursor = nl + 1;
            }
        }
        if (!pending.empty()) aggregator.consume(pending);
        if (gzclose(input) != Z_OK) throw std::runtime_error("gzip close failed");
        if (!aggregator.saw_header || aggregator.seen_symbols != axis)
            throw std::runtime_error("not all predeclared gene symbols found in raw count file");
        aggregator.write_library();
        aggregator.matrix.close();
        aggregator.genes.close();
        aggregator.library.close();
        if (!aggregator.matrix || !aggregator.genes || !aggregator.library)
            throw std::runtime_error("aggregate close failed");
        fs::rename(matrix_part, matrix_final);
        fs::rename(genes_part, genes_final);
        fs::rename(library_part, library_final);
        std::cout << "{\"stage\":\"E258_RAW_ALLOWED_GROUP_SUMS\","
                  << "\"gene_rows_seen\":" << aggregator.rows_seen << ","
                  << "\"gene_rows_kept\":" << aggregator.rows_kept << ","
                  << "\"selected_numeric_tokens_parsed\":" << aggregator.selected_tokens << ","
                  << "\"skipped_numeric_tokens_not_parsed\":" << aggregator.skipped_numeric_tokens << ","
                  << "\"test_target_numeric_tokens_parsed\":0}"
                  << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "E258 raw aggregation failed: " << error.what() << std::endl;
        return 1;
    }
}
