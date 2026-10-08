using RentalQA.Core.Models;

namespace RentalQA.Core.Repositories;

public class InMemoryRentalRepository : IRentalRepository
{
    private readonly List<RentalRecord> _records = new();
    private int _nextId = 1;

    public RentalRecord Add(RentalRecord record)
    {
        record.Id = _nextId++;
        _records.Add(record);
        return record;
    }

    public RentalRecord? GetById(int id) =>
        _records.FirstOrDefault(r => r.Id == id);

    public IEnumerable<RentalRecord> GetAll() => _records;
}
