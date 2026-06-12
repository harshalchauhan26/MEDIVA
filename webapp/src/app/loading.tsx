export default function Loading() {
  return (
    <div className="space-y-6">
      <div className="h-8 w-48 skeleton" />
      <div className="grid gap-4 sm:grid-cols-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="card p-5">
            <div className="h-4 w-24 skeleton" />
            <div className="mt-3 h-8 w-16 skeleton" />
          </div>
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <div className="card h-64 p-5 lg:col-span-2">
          <div className="h-4 w-40 skeleton" />
        </div>
        <div className="card h-64 p-5">
          <div className="h-4 w-32 skeleton" />
        </div>
      </div>
    </div>
  );
}
