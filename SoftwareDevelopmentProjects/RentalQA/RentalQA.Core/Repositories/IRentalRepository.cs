using RentalQA.Core.Models;

namespace RentalQA.Core.Repositories;

/// <summary>
/// Abstraction over the data store. In production this would be implemented
/// with EF Core against SQL Server; for this demo it's backed by an in-memory list.
/// Talking point: this separation is what lets you write fast unit tests against
/// the in-memory version AND real backend/SQL integration tests against the
/// production implementation, without changing a single line of business logic.
/// </summary>
public interface IRentalRepository
{
    RentalRecord Add(RentalRecord record);
    RentalRecord? GetById(int id);
    IEnumerable<RentalRecord> GetAll();
}
