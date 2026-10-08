# RentalQA — Automated Testing Portfolio Project

A small rental-pricing Web API built specifically to demonstrate the four testing
skills: **unit testing**,
**API testing**, **backend/data verification**, and **automation-framework design**
on the **Microsoft stack** (C#, ASP.NET Core Web API, xUnit, Azure DevOps).

## What's in here

```
RentalQA/
├── RentalQA.Core/          # Business logic: pricing, mileage fees, late fees
├── RentalQA.Api/           # Minimal ASP.NET Core Web API exposing /rentals
├── RentalQA.Tests/         # xUnit unit tests + API integration tests
├── azure-pipelines.yml     # CI pipeline: build, test, publish results
└── RentalQA.sln
```

## Why it's built this way 

- **`IRentalRepository`** is an interface, not a concrete class. The demo uses
  an in-memory implementation, but in production this would be EF Core against
  SQL Server — swapping the backend requires changing **one line** in `Program.cs`,
  not rewriting the business logic or the tests. This is the same pattern that
  lets you write fast unit tests against an in-memory store and slower, real
  integration tests against an actual database, without duplicating test logic.
- **Money is `decimal`, never `float`/`double`** — a deliberate detail. Floating-point
  rounding errors in financial calculations are a classic, avoidable bug.
- **Late fee logic rounds any partial day up to a full late day** (`Math.Ceiling`)
  — a real business rule that's easy to get subtly wrong with naive date math,
  and a good thing to explain if asked "walk me through a tricky test case."
- **Tests are split into two files on purpose**: `RentalPriceCalculatorTests.cs`
  tests business logic directly (fast, no HTTP), while `RentalsApiTests.cs` uses
  `WebApplicationFactory` to spin up the real API and hit it over HTTP — this is
  the actual difference between a *unit test* and an *API/integration test*,
  and being able to explain that distinction clearly is one of the more common
  QA interview questions.

## How to run it

```bash
# From the RentalQA/ folder:
dotnet restore
dotnet build
dotnet test                     # runs all unit + API tests
dotnet run --project RentalQA.Api   # starts the API at https://localhost:xxxx/swagger
```

Swagger UI will be available at `/swagger` once the API is running, so you can
manually exercise the endpoints — useful if you're asked "how would you test
this manually before automating it."

## Extending it 

1. Add a real `SqlServerRentalRepository : IRentalRepository` using Entity
   Framework Core, and re-run the same test suite against it to prove backend
   parity.
2. Add a GitHub Actions or Azure Pipelines badge showing tests passing on every
   push — turns this into a live, provable CI signal rather than a one-time run.
3. Add a `POST /rentals/{id}/return` endpoint that records the actual return
   date, which would let you test the late-fee logic end-to-end through the API
   rather than only at the unit level.
4. Add a basic Playwright or Selenium UI test against the Swagger page, since
   the JD also calls out UI testing — this would round out all three testing
   layers (unit, API, UI) in one portfolio project.


